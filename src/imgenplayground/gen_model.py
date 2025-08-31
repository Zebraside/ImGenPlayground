import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils.torch_utils import is_compiled_module


def unwrap_model(model):
    model = model._orig_mod if is_compiled_module(model) else model
    return model


class GenModel(torch.nn.Module):
    def __init__(self, pretrained_model_name_or_path): # , revision, variant, non_ema_revision
        super().__init__()

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        # self.revision = revision
        # self.variant = variant
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
        # self.unet = torch.compile(self.unet)
        # self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        # self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        # self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")

    def init_pipeline(self, device, weight_dtype=torch.bfloat16, xformers_enabled=False):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.pretrained_model_name_or_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(self.pretrained_model_name_or_path, subfolder="vae")

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            vae=unwrap_model(self.vae),
            text_encoder=(self.text_encoder),
            tokenizer=self.tokenizer,
            unet=unwrap_model(self.unet),
            safety_checker=None,
            # revision=self.revision,
            # variant=self.variant,
            torch_dtype=weight_dtype,
        )

        self.pipeline = self.pipeline.to(device)
        if xformers_enabled:
            self.pipeline.enable_xformers_memory_efficient_attention()

    def cleanup_pipeline(self):
        del self.pipeline
        del self.tokenizer
        del self.text_encoder
        del self.vae

    def generate(self, prompts, prompt_embeds=None):
        assert self.pipeline is not None
        # TODO: add height, width
        return self.pipeline(prompt=prompts, width=256, height=256, prompt_embeds=prompt_embeds)[0]

    def enbale_ema(self):
        # TODO: Implement
        pass
