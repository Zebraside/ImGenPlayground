import torch
import lightning as L
import torch.nn.functional as F

from imgenplayground.gen_model import GenModel

class LitImageGen(L.LightningModule):
    def __init__(self, gen_model: GenModel,
                       lr=1e-6,
                       min_lr=0.0,
                       betas=(0.9, 0.999),
                       weight_decay=1e-2,
                       adam_epsilon=1e-08,
                       warmup_steps=50,
                       max_steps=3000,
                       enable_gradient_checkpointing=False,
                       allow_tf32=False,
                       scale_lr=False):
        super().__init__()
        self.save_hyperparameters(ignore=["gen_mdoel"])
        self.gen_model = gen_model
        # gen_model.vae.requires_grad_(False)
        # gen_model.text_encoder.requires_grad_(False)

        # TODO: enable xformers for unet
        if enable_gradient_checkpointing:
            self.gen_model.unet.enable_gradient_checkpointing()

        # TODO: benchmark it
        if allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # TODO: compile unet
        #
        # TODO: implement
        if scale_lr:
            # args.learning_rate = (
            #     args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
            # )
            pass

    def training_step(self, batch, batch_idx):
        latents = batch["pixel_values"] # image is preprocessed
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # TODO: noise offset?
        # TODO: input_perturbation?
        #
        timesteps = torch.randint(0, self.gen_model.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        noisy_latents = self.gen_model.noise_scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = batch["encoder_hidden_states"] # text is preprocessed
        target = noise # TODO: other predicion type?

        model_pred = self.gen_model.unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return loss

    def on_validation_epoch_start(self):
        self.gen_model.init_pipeline(self.device)

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        self.gen_model.cleanup_pipeline()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                     lr=self.hparams.lr,
                                     betas=self.hparams.betas,
                                     weight_decay=self.hparams.weight_decay,
                                     eps=self.hparams.adam_epsilon)
        schedulers = [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.hparams.warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_steps - self.hparams.warmup_steps, eta_min=self.hparams.min_lr)
        ]
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=schedulers, milestones=[self.hparams.warmup_steps])
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
