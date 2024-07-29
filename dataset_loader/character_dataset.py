import string
import random

import torch
from torch.utils.data import Dataset

"""
Loads Character's Embeddings Dataset.
"""
class CharacterEmbeddingDataset(Dataset):
    def __init__(
            self,
            dataset,
            padding_idx,
            context_window=200):
        self.dataset = dataset
        self.padding_idx = padding_idx
        self.context_window = context_window

        if len(self.dataset) == 0:
            raise Exception("No data found.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Ensures words are not cut off when sampling their characters indices.
        combined_data = []
        for i in range(index, len(self.dataset)):
            combined_data.extend(self.dataset[i])

            if len(combined_data) > self.context_window:
                break

        if len(combined_data) > self.context_window:
            input_data = combined_data[0:self.context_window]
            target_data = combined_data[1:self.context_window+1]
        else:
            input_data = combined_data[0:len(combined_data)-1]
            target_data = combined_data[1:len(combined_data)]

            # Integer index for Pad token, length of vocabulary list.
            input_pad_index = [self.padding_idx] * (self.context_window - len(input_data))
            target_pad_index = [self.padding_idx] * (self.context_window - len(target_data))

            input_data.extend(input_pad_index)
            target_data.extend(target_pad_index)

        input_tensor = torch.tensor(input_data)
        target_tensor = torch.tensor(target_data)

        return input_tensor, target_tensor
