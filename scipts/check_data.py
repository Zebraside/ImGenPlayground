import torch
from imgenplayground.gen_model import GenModel

# import polars as pl

# df = pl.read_parquet('naruto.parquet')

# from PIL import Image
# import io
# Image.open(io.BytesIO(df[0]["image"][0]["bytes"])).save("image.png")
# df.write_parquet("naruto.parquet")

# sample_model = torch.nn.Linear(512, 512)
# optimizer = torch.optim.AdamW(sample_model.parameters())
# schedulers = [
#     torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=300),
#     torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000 - 300, eta_min=1e-7)
# ]
# scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=schedulers, milestones=[300])

# lrs = []
# for i in range(50000):
#     lrs.extend(scheduler.get_last_lr())
#     scheduler.step()

# print(lrs)
# import matplotlib.pyplot as plt
# plt.plot(lrs)
# plt.show()

model = GenModel("stabilityai/stable-diffusion-2")
model.init_pipeline("mps", weight_dtype=torch.float16)

model.generate("A man")
