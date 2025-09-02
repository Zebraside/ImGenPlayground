import lightning as L
import torch.utils.data as data
import torch 
import lovely_tensors as lt
lt.monkey_patch()
from icecream import install
install()
from lightning.pytorch.loggers import WandbLogger


from imgenplayground.gen_model import GenModel
from imgenplayground.lit_module import LitImageGen
from imgenplayground.datasets import NarutoDataset

torch.set_float32_matmul_precision('high')

dataset = NarutoDataset("naruto.parquet", "stabilityai/stable-diffusion-2")
train, val = data.random_split(dataset, [len(dataset) - 1, 1])

autoencoder = LitImageGen(
    GenModel("stabilityai/stable-diffusion-2"),
    lr=1e-4,
    warmup_steps=500,
    allow_tf32=True
)

logger =  WandbLogger(project="ImGenPlayground")

trainer = L.Trainer(
    precision="bf16-mixed",
    max_steps=5000,
    logger=logger
)
trainer.fit(autoencoder, 
            data.DataLoader(train, num_workers=8, batch_size=12, shuffle=True), 
            data.DataLoader(val))
