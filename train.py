import lightning as L
import torch.utils.data as data

from imgenplayground.gen_model import GenModel
from imgenplayground.lit_module import LitImageGen
from imgenplayground.datasets import NarutoDataset

dataset = NarutoDataset("/Users/kmolchan/Dev/ImGenPlayground/naruto.pickle")
train, val = data.random_split(dataset, [len(dataset) - 1, 1])

autoencoder = LitImageGen(
    GenModel("stabilityai/stable-diffusion-2")
)

trainer = L.Trainer(
    precision="bf16-mixed"
)
trainer.fit(autoencoder, data.DataLoader(train), data.DataLoader(val))
