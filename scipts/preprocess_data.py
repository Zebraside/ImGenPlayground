from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from torchvision import transforms as tr
import torch
import time
import polars as pl
import io
import tqdm
import pickle

transforms = tr.Compose([
    tr.Resize(256),
    tr.ToTensor()
])

vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2", subfolder="vae")
tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="text_encoder")
vae.eval()
text_encoder.eval()

device="mps"
dtype=torch.float32
vae.to(device, dtype=dtype)
text_encoder.to(device)

data = pl.read_parquet("/Users/kmolchan/Dev/ImGenPlayground/naruto.parquet")
tasks = []
with torch.inference_mode():
    for task in tqdm.tqdm(data.iter_rows(named=True), total=len(data)):
        # print(task)
        image = Image.open(io.BytesIO(task["image"]["bytes"]))
        image = transforms(image).to(device, dtype=dtype)
        output = (vae.encode(image[None]).latent_dist.sample() * vae.config.scaling_factor)[0].cpu()

        tokenized_text = tokenizer.encode(task["text"])
        encoded_text = text_encoder(torch.tensor(tokenized_text)[None].to(device), return_dict=False)[0][0].cpu()

        tasks.append({
            "pixel_values": output,
            "encoder_hidden_states": encoded_text
        })

with open("naruto.pickle", "wb") as f:
    pickle.dump(tasks, f)
