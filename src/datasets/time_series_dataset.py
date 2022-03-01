import numpy as np
import torch
from torch.utils.data import Dataset


class TimeseriesDataset(Dataset):
    """
    Custom Dataset subclass.
    Serves as input to DataLoader to transform X
      into sequence data using rolling window.
    DataLoader using this dataset will output batches
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs.
    """

    def __init__(self, time_series: np.ndarray, seq_len: int, y_size: int):
        self.time_series = torch.tensor(time_series).float()
        self.seq_len = seq_len
        self.y_size = y_size

    def __len__(self) -> int:
        return self.time_series.__len__() - (self.seq_len + self.y_size - 1)

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        # return x, y
        return (
            self.time_series[index : index + self.seq_len],
            self.time_series[index + self.seq_len + self.y_size - 1],
        )
