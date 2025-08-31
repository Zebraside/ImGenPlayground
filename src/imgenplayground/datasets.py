import torch
# import polars
import pickle


class NarutoDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_preprocessed):
        super().__init__()
        with open(path_to_preprocessed, "rb") as f:
            self.dataset = pickle.load(f)

    def __getitem__(self, idx):
        return {
            "pixel_values": self.dataset[idx]["pixel_values"],
            "encoder_hidden_states": self.dataset[idx]["encoder_hidden_states"],
        }

    def __len__(self):
        return len(self.dataset)
