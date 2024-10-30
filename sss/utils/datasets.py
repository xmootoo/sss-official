import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tslearn.utils import to_time_series_dataset

class ForecastingDataset(Dataset):
    """
    A standard forecasting dataset class for PyTorch.

    Args:
        data (torch.Tensor): The time series data in a tensor of shape (num_channels, num_time_steps).
        seq_len (int): The length of the input window.
        pred_len (int): The length of the forecast window.

    __getitem__ method: Returns the input and target data for a given index, where the target window follows
                        immediately after the input window in the time series.

    """

    def __init__(self, data, seq_len, pred_len, dtype="float32"):

        # Convert the data to a tensor and set the datatype
        dtype = eval("torch." + dtype)
        if not torch.is_tensor(data):
            self.data = torch.from_numpy(data).type(dtype)
        else:
            self.data = data.type(dtype)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return self.data.shape[1] - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        input_data = self.data[:, idx:idx+self.seq_len]
        target_data = self.data[:, idx+self.seq_len:idx+self.seq_len+self.pred_len]
        return input_data, target_data


class UnivariateForecastingDataset(Dataset):
    """
    A standard forecasting dataset class for PyTorch.

    Args:
        data_x (torch.Tensor): The time series data in a tensor of shape (num_windows, seq_len).
        data_y (torch.Tensor): The time series data in a tensor of shape (num_windows, pred_len).

    __getitem__ method: Returns the input and target data for a given index, where the target window follows
                        immediately after the input window in the time series.

    """

    def __init__(self, x, y, dtype="float32"):

        # Convert the data to a tensor and set the datatype
        dtype = eval("torch." + dtype)
        if not torch.is_tensor(x):
            self.x = torch.from_numpy(x).type(dtype)
            self.y = torch.from_numpy(y).type(dtype)
        else:
            self.x = x.type(dtype)
            self.y = y.type(dtype)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ClassificationDataset(Dataset):
    """
    A classification dataset class for time series.

    Args:
        x (torch.Tensor): The input data in a tensor of shape (num_windows, seq_len).
        y (torch.Tensor): The target data in a tensor of shape (num_windows).
        ch_ids (torch.Tensor): The channel IDs in a tensor of shape (num_windows).
        t (torch.Tensor): The time indices in a tensor of shape (num_windows).

    """

    def __init__(self, x, y, ch_ids=None, task="binary", full_channels=False):

        # Data
        self.x = x
        self.y = torch.tensor(y) if isinstance(y, list) else y
        ch_ids = torch.tensor(ch_ids) if isinstance(ch_ids, list) else ch_ids
        self.ch_ids = ch_ids
        self.full_channels = full_channels


        # Parameters
        self.task = task
        self.len = x.size(0)
        self.num_classes = len(torch.unique(self.y))

        # Channel IDs
        if not full_channels:
            self.unique_ch_ids = torch.unique(ch_ids, sorted=True).tolist()
            label_indices = torch.tensor([torch.where(ch_ids == unique_id)[0][0] for unique_id in self.unique_ch_ids])
            self.unique_ch_labels = self.y[label_indices].tolist()
            self.ch_labels = dict()
            for i, ch_id in enumerate(self.unique_ch_ids):
                self.ch_labels[ch_id] = int(self.unique_ch_labels[i])

        if ch_ids is not None and not full_channels:
            unique_ch_ids, indices = torch.unique(ch_ids, sorted=True, return_inverse=True)

            self.ch_id_list = unique_ch_ids.tolist()
            self.num_channels = len(unique_ch_ids)
            self.ch_targets = torch.zeros(self.num_channels)

            # Get the unique labels for each channel
            for i, ch_id in enumerate(unique_ch_ids):
                matching_indices = torch.where(ch_ids == ch_id)
                label = self.y[matching_indices][0]
                self.ch_targets[i] = label

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        In a dataloader it returns appropriate tensors for CrossEntropy loss.
            x: (batch_size, 1, seq_len)
            y: (batch_size,)
            ch_ids: (batch_size,)
        """

        output = []

        if self.task=="multi":
            label = self.y[idx].long()
        elif self.task=="binary":
            label = self.y[idx].float()
        else:
            raise ValueError("Task must be either 'binary' or 'multi'.")

        if self.ch_ids is not None and not self.full_channels:
            output += [self.x[idx].unsqueeze(0), label, self.ch_ids[idx]]
        else:
            output += [self.x[idx].unsqueeze(0), label]

        return tuple(output)

class VariableLengthDataset(Dataset):
    def __init__(self, x, y, pad_to_max=False, batch_size=1, tslearn=False, numpy_data=False, dtype="float32", task="binary"):
        """
        Dataset for variable-length sequences.

        Args:
        data (list): List of 1D numpy arrays or torch tensors
        pad_to_max (bool): If True, pad all sequences to the length of the longest sequence
        """

        # Padding must be enabled for multi-signal batching
        if batch_size>1:
            assert pad_to_max==True, "VariableLengthDataset only supports batch size of 1 for unpadded signals."

        # Data
        self.x = x
        self.y =  y
        self.pad_to_max = pad_to_max
        self.numpy_data = numpy_data
        self.tslearn = tslearn
        self.torch_dtype = getattr(torch, dtype)
        self.np_dtype = np.dtype(dtype)
        self.task = task

        # Find the max and min sequence lengths
        lengths = [len(seq.squeeze()) for seq in self.x]

        if len(lengths)==0:
            self.max_length = self.min_length = 0
        else:
            self.max_length = max(lengths); self.min_length = min(lengths)
        print(f"Max length sequence: {self.max_length}. Min length sequence: {self.min_length}")

        # Convert all sequences to tensors and dtype (Optional)
        # Convert the string to actual torch and numpy dtypes

        # Convert the data
        self.x = [torch.from_numpy(seq.astype(self.np_dtype)) if isinstance(seq, np.ndarray)
                    else seq.to(self.torch_dtype) if isinstance(seq, torch.Tensor)
                    else torch.tensor(seq, dtype=self.torch_dtype)
                    for seq in self.x]


        # Pad sequences to max length (Optional. Only for torch tensors)
        if pad_to_max and not numpy_data:
            self.x = [self._pad_sequence(seq) for seq in self.x]

        # Convert to tslearn dataset (Optional)
        if tslearn:
            self.x = [seq.squeeze() for seq in self.x]
            self.x = to_time_series_dataset(self.x)

        # Convert targets to list -> numpy array if not already
        if isinstance(self.y, list):
            if self.numpy_data:
                self.y = np.array(self.y, dtype=self.np_dtype)
            else:
                self.y = torch.tensor(self.y, dtype=self.torch_dtype)



    def _pad_sequence(self, sequence):
        """Pad a sequence to max_length with zeros."""
        padding_length = self.max_length - len(sequence)
        return torch.nn.functional.pad(sequence, (0, padding_length), 'constant', 0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        signal = self.x[idx].squeeze() if not self.tslearn else self.x[idx]
        label = self.y[idx].long() if self.task=="multi" else self.y[idx].float()
        return signal, label, len(signal)  # Return both the sequence and its original length


if __name__=="__main__":


    # VariableLengthDataset
    # Example usage:
    seq1 = np.array([1, 2, 3])
    seq2 = np.array([4, 5, 6, 7, 8])
    seq3 = torch.tensor([9, 10])

    # Without padding
    dataset = VariableLengthDataset([seq1, seq2, seq3], [0,1,0])
    print("Dataset without padding:")
    for i in range(len(dataset)):
        sequence, label, length = dataset[i]
        print(f"Sequence {i+1}: {sequence}, Original length: {length}")
    print(f"Max length: {dataset.max_length}")

    # With padding
    padded_dataset = VariableLengthDataset([seq1, seq2, seq3], [0,1,0], pad_to_max=True)
    print("\nDataset with padding:")
    for i in range(len(padded_dataset)):
        sequence, label, length = padded_dataset[i]
        print(f"Sequence {i+1}: {sequence}, Original length: {length}")
    print(f"Max length: {padded_dataset.max_length}")
