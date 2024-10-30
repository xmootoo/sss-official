import os
import gc
from itertools import chain
import pickle
import random
from collections import Counter
import statistics as stats

import torch
from torch.functional import _return_counts
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

# ROCKET
from sss.layers.rocket.random_kernels import generate_kernels, apply_kernels

# UEA
# import aeon
# from aeon.datasets import load_from_tsfile

# Dataset Classes
from sss.utils.datasets import ForecastingDataset, UnivariateForecastingDataset, ClassificationDataset, VariableLengthDataset

# Normalization, Preprocessing, Dataloading
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from scipy import interpolate

#<-----------------Forecasting------------------------>
def load_forecasting(
    dataset_name="ETTh1",
    seq_len=512,
    pred_len=96,
    window_stride=1,
    scale=True,
    train_split=0.7,
    val_split=0.1,
    univariate=False,
    resizing_mode="None",
    target_channel=-1):
    """
    Args:
        dataset_name (str): Name of the dataset to load. Options: "ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic", "illness".
        seq_len (int): Length of the input sequence.
        pred_len (int): Length of the prediction sequence.
        window_stride (int): Stride of the window for sliding window sampling (if univariate is True).
        scale (bool): Whether to normalize the data.
        train_split (float): Fraction of the data to use for training.
        val_split (float): Fraction of the data to use for validation.
        univariate (bool): Whether to use univariate or multivariate data, this processes the data differently and creates windows before
                           being input into the dataloader, so as to allow appropriate separation between the channels.
    Returns:
        train_data (np.array): Training data of shape (num_channels, train_len).
        val_data (np.array): Validation data of shape (num_channels, val_len).
        test_data (np.array): Test data of shape (num_channels, test_len).
    """

    # Load data. NumPy array of shape (seq_len, num_channels)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))

    if dataset_name == "rf_emf":
        data = loadmat(os.path.join(root_dir, "data/forecasting/rf_emf/LongTerm17.mat"))
        data = data["York_LongTerm17"].T

    elif dataset_name == "rf_emf_det":
        data = pd.read_csv(os.path.join(root_dir, "data/forecasting/rf_emf2/DET_Monitoring.csv")).to_numpy()
    elif dataset_name == "rf_emf_ptv":
        data = pd.read_csv(os.path.join(root_dir, "data/forecasting/rf_emf2/PTV_Monitoring.csv")).to_numpy()
    else:
        data = pd.read_csv(os.path.join(root_dir, "data/forecasting", f"{dataset_name}.csv"))
        data = data.drop(columns=["date"]).values

    # Define train, validation, and test indices
    if "etth" in dataset_name.lower():
        train_idx = [0, 12 * 30 * 24]
        val_idx = [12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24]
        test_idx = [12 * 30 * 24 + 4 * 30 * 24 - seq_len, 12 * 30 * 24 + 8 * 30 * 24]
    elif "ettm" in dataset_name.lower():
        train_idx = [0, 12 * 30 * 24 * 4]
        val_idx = [12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4]
        test_idx = [12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    else:
        test_split = 1 - train_split - val_split
        num_train = int(len(data) * train_split)
        num_test = int(len(data) * test_split)
        num_val = len(data) - num_train - num_test

        train_idx = [0, num_train]
        val_idx = [num_train - seq_len, num_train + num_val]
        test_idx = [len(data) - num_test - seq_len, len(data)]

    # Split data
    train_data = data[train_idx[0]:train_idx[1]]
    val_data = data[val_idx[0]:val_idx[1]]
    test_data = data[test_idx[0]:test_idx[1]]

    # Normalization. Data must be in shape (seq_len, num_channels)
    if scale:
        scaler = StandardScaler()

        scaler.fit(train_data)

        train_data, val_data, test_data = (
            scaler.transform(train_data),
            scaler.transform(val_data),
            scaler.transform(test_data)
        )

    # Univariate. Create windows directly (separated by channel), and return input and labels separately
    if univariate:
        channels = range(data.shape[1]) if target_channel < 0 else [target_channel]
        train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels = [], [], [], [], [], []
        for i in channels:
            train_window = create_windows(torch.from_numpy(train_data.T[i]), seq_len+pred_len, window_stride, resizing_mode)
            val_window = create_windows(torch.from_numpy(val_data.T[i]), seq_len+pred_len, window_stride, resizing_mode)
            test_window = create_windows(torch.from_numpy(test_data.T[i]), seq_len+pred_len, window_stride, resizing_mode)

            train_x = train_window[:, :seq_len]; train_y = train_window[:, seq_len:]
            val_x = val_window[:, :seq_len]; val_y = val_window[:, seq_len:]
            test_x = test_window[:, :seq_len]; test_y = test_window[:, seq_len:]

            train_inputs.append(train_x); train_labels.append(train_y);
            val_inputs.append(val_x); val_labels.append(val_y);
            test_inputs.append(test_x); test_labels.append(test_y)

        train_data = torch.cat(train_inputs, dim=0)
        train_labels = torch.cat(train_labels, dim=0)
        val_data = torch.cat(val_inputs, dim=0)
        val_labels = torch.cat(val_labels, dim=0)
        test_data = torch.cat(test_inputs, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

        return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)

    return train_data.T, val_data.T, test_data.T

def load_splits(data, train_split=0.6, val_split=0.2, scale=True):
    """
    Loads the training, validation, and test splits of the data tensor.

    Args:
        data (numpy.ndarray): The time series data in a tensor of shape (num_time_steps, num_channels).
        train_split (float): The proportion of the data to use for training.
        val_split (float): The proportion of the data to use for validation.
    Returns:
        numpy.ndarray: The training split of the data tensor (num_channels, train_len).
        numpy.ndarray: The validation split of the data tensor (num_channels, val_len).
        numpy.narray: The test split of the data tensor (num_channels, test_len).
    """

    num_time_steps = data.shape[0]
    train_slice = slice(None, int(train_split * num_time_steps))
    val_slice = slice(int(train_split * num_time_steps), int((train_split + val_split) * num_time_steps))
    test_slice = slice(int((train_split + val_split) * num_time_steps), None)
    train_data, val_data, test_data = data[train_slice, :], data[val_slice, :], data[test_slice, :]

    if scale:
        scaler = StandardScaler()
        scaler.fit(train_data)
        train_data, val_data, test_data = scaler.transform(train_data), scaler.transform(val_data), scaler.transform(test_data)

    return train_data.T, val_data.T, test_data.T


#<-----------------OpenNeuro SOZ Classification------------------------>
def create_windows(tensor, window_size, window_stride, resizing_mode="None"):
    """
    Uses the sliding window technique to sample windows for a time series tensor.

    Args:
        tensor (torch.Tensor): The input tensor of shape (seq_len, num_channels).
        window_size (int): The size of the window to sample.
        window_stride (int): The stride of the window.
    """
    seq_len = tensor.size(0)
    if window_size == -1:
            return tensor.unsqueeze(0) # Return the entire tensor as a single window
    elif window_size > seq_len and resizing_mode not in {"pad_trunc", "resize"}:
        raise ValueError("Window size must be less than or equal to the sequence length")
    elif window_size > seq_len and resizing_mode in {"pad_trunc", "resize"}:
        data = [tensor]
        return resize_sequence(data, window_size, resizing_mode)

    if window_stride == -1:
        window_stride = window_size # Set window stride equal to window stride for non-overlapping sampling

    num_windows = (seq_len - window_size) // window_stride + 1  # General formula for any stride value
    if num_windows <= 0:
        raise ValueError("No windows can be formed with the given parameters")

    return torch.as_strided(tensor, size=(num_windows, window_size), stride=(window_stride, 1))

def create_equidistant_windows(time_series, num_windows, window_length):
    """
    Sample equidistant windows from a multi-channel time series tensor.

    This function samples 'num_windows' windows of length 'window_length' from the input 'time_series'.
    The windows are as equidistant as possible given the constraints of the input length and desired window parameters.

    Args:
        time_series (torch.Tensor): Input tensor of shape (sequence_length, num_channels).
        num_windows (int): Number of windows to sample.
        window_length (int): Length of each window.

    Returns:
        torch.Tensor: Tensor of sampled windows with shape (num_windows, window_length, num_channels).

    Raises:
        ValueError: If the input tensor does not have 2 dimensions.

    Notes:
        - If sequence_length < window_length, the time series is padded to window_length and returned as a single window.
        - If num_windows == 1, the center window is returned.
        - If sequence_length/window_length < num_windows, windows will overlap to provide the requested number of windows.
    """
    # Check input dimensions
    if time_series.dim() != 2:
        raise ValueError("Input tensor must have shape (sequence_length, num_channels)")

    L, C = time_series.shape
    N = num_windows
    W = window_length

    # Handle the case where L < W
    if L < W:
        # Pad the time series to length W
        padded_series = torch.nn.functional.pad(time_series, (0, 0, 0, W - L))
        return padded_series.unsqueeze(0)  # Return shape (1, W, C)

    if N == 1:
        # If only one window is requested, return the center window
        start = (L - W) // 2
        return time_series[start:start+W].unsqueeze(0)

    # Calculate the step size between window starts
    if L - W >= N - 1:
        # No overlap needed
        step = (L - W) // (N - 1)
    else:
        # Overlap needed
        step = (L - W) / (N - 1)

    # Calculate the start indices of each window
    if isinstance(step, int):
        start_indices = torch.arange(0, L - W + 1, step)[:N]
    else:
        start_indices = torch.linspace(0, L - W, N).long()

    # Sample the windows
    windows = torch.stack([time_series[i:i+W] for i in start_indices])

    return windows

def resize_sequence(data, max_seq_len=3000, resizing_mode="none"):
    if not isinstance(data, list):
        raise ValueError("Input must be a list of 1D numpy arrays.")

    resized_data = np.zeros((len(data), max_seq_len))

    for i, seq in enumerate(data):
        if isinstance(seq, torch.Tensor):
            seq = seq.numpy().squeeze()
        if not isinstance(seq, np.ndarray) or seq.ndim != 1:
            raise ValueError("Each element in the list must be a 1D numpy array.")

        if resizing_mode == "pad_trunc":
            if seq.shape[0] >= max_seq_len:
                resized_data[i] = seq[:max_seq_len]
            else:
                resized_data[i, :seq.shape[0]] = seq

        elif resizing_mode == "resize":
            if seq.shape[0] == 1:
                resized_data[i] = np.full(max_seq_len, seq[0])
            else:
                x = np.linspace(0, 1, seq.shape[0])
                f = interpolate.interp1d(x, seq)
                x_new = np.linspace(0, 1, max_seq_len)
                resized_data[i] = f(x_new)

        else:
            raise ValueError("Invalid resizing mode. Choose from 'pad_trunc' or 'resize'.")

    return torch.from_numpy(resized_data)

def load_open_neuro_interchannel(patient_cluster="jh",
                        kernel_size=150,
                        kernel_stride=75,
                        window_size=512,
                        window_stride=24,
                        dtype="float32",
                        pool_type="avg",
                        balance=True,
                        scale=True,
                        train_split=0.6,
                        val_split=0.2,
                        seed=1995,
                        task="binary",
                        full_channels=False,
                        multicluster=True,
                        resizing_mode="None",
                        median_seq_len=False,
                        median_seq_only=False):

        """
        Splits train, validation, and test sets by allocating them separate channels. Then from those channels
        creates windows of the specified size and stride. Finally, pools the windows using the specified pooling
        type (avg or max) with a kernel size and stride. Returns the train, validation, and test sets as well as
        the corresponding labels.

        Patient cluster legend:
            "pt": "National Institute of Health",
            "ummc": "University of Maryland Medical Center",
            "jhh": "Johns Hopkins Hospital",
            "umf": "University of Miami Florida Hospital",

        Args:
            patient_cluster (str): The patient cluster to load the data from. Options: "jh", "pt", "umf", "ummc".
            kernel_size (int): The size of the pooling kernel.
            kernel_stride (int): The stride of the pooling kernel.
            window_size (int): The size of the window.
            window_stride (int): The stride of the window.
            pool_type (str): The type of pooling to use. Options: "avg", "max".
            balance (bool): Whether to balance the classes within train, validation, and test sets. Balancing is done by channel labels.
            scale (bool): Whether to normalize the data along the channel dimension.
            train_split (float): The proportion of the data to use for training.
            val_split (float): The proportion of the data to use for validation.
            seed (int): The random seed to use for reproducibility.
            task (str): The task to perform. Options: "binary" or "multi". Binary is the task of classifying each channel as SOZ or non-SOZ.
                        whereas multiclass is the task of classifying each channel and patient outcome:
                        #   0 - no SOZ, positive outcome
                        #   1 - SOZ, positive outcome
                        #   2 - no SOZ, negative outcome
                        #   3 - SOZ, negative outcome
                        only available for "pt" and "ummc" clusters, as "jh" and "umf" have only labels {2,3} and {0,1} respectively.
        Returns:
            train_data, val_data, test_data: The train, validation, and test data tensors each of shape (num_windows, seq_len)
            train_labels, val_labels, test_labels: The train, validation, and test labels tensors each of shape (num_windows,).
            train_ch_ids, val_ch_ids, test_ch_ids: The train, validation, and test channel IDs each of shape (num_windows,).
        """
        binary_clusters = {"jh", "umf", "pt", "ummc"}
        multi_clusters = {"pt", "ummc"}
        test_split = 1 - train_split - val_split

        dtype = torch.float32 if dtype == "float32" else torch.float64

        # Assert the correct patient clusters for multiclass classification
        if task=="multi": assert patient_cluster in multi_clusters, f"Invalid patient cluster '{patient_cluster}' (multiclass classification). Options: {multi_clusters}."

        # Set seed
        random.seed(seed)

        # Single signals
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(current_dir))

        # Load the data
        data_folder = "single_signals" if task == "binary" else "single_signals_multi"
        with open(os.path.join(root_dir, f"data/classification/open_neuro/{data_folder}", f"signals_{patient_cluster}.pkl"), "rb") as f:
            data = pickle.load(f)

        # Number of channels
        num_channels = len(data)

        # Class balancing
        if balance and task == "binary":
            labels = [data[i][1] for i in range(len(data))]
            zero_indices = [i for i, x in enumerate(labels) if x == 0]
            one_indices = [i for i, x in enumerate(labels) if x == 1]
            min_counts = min(len(zero_indices), len(one_indices))
            min_counts = min_counts // 4 if patient_cluster=="pt" else min_counts #TODO: Change this to be dynamic later

            zero_indices = random.sample(zero_indices, min_counts)
            one_indices = random.sample(one_indices, min_counts)

            # Train, val, test split (class balanced indices)
            train_indices = zero_indices[:int(train_split*min_counts)] + one_indices[:int(train_split*min_counts)]
            val_indices = zero_indices[int(train_split*min_counts):int((train_split+val_split)*min_counts)] + one_indices[int(train_split*min_counts):int((train_split+val_split)*min_counts)]
            test_indices = zero_indices[int((train_split+val_split)*min_counts):] + one_indices[int((train_split+val_split)*min_counts):]

            train_channels = [[data[i], i] for i in train_indices]
            val_channels = [[data[i], i] for i in val_indices]
            test_channels = [[data[i], i] for i in test_indices]
        elif balance and task == "multi":
            labels = [data[i][1] for i in range(len(data))]
            zero_indices = [i for i, x in enumerate(labels) if x == 0]
            one_indices = [i for i, x in enumerate(labels) if x == 1]
            two_indices = [i for i, x in enumerate(labels) if x == 2]
            three_indices = [i for i, x in enumerate(labels) if x == 3]

            min_counts = min(len(zero_indices), len(one_indices), len(two_indices), len(three_indices))
            min_counts = min_counts // 4 if patient_cluster=="pt" else min_counts #TODO: Change this to be dynamic later

            zero_indices = random.sample(zero_indices, min_counts)
            one_indices = random.sample(one_indices, min_counts)
            two_indices = random.sample(two_indices, min_counts)
            three_indices = random.sample(three_indices, min_counts)

            # Train, val, test split (class balanced indices)
            train_indices = zero_indices[:int(train_split*min_counts)] + one_indices[:int(train_split*min_counts)] + two_indices[:int(train_split*min_counts)] + three_indices[:int(train_split*min_counts)]
            val_indices = zero_indices[int(train_split*min_counts):int((train_split+val_split)*min_counts)] + one_indices[int(train_split*min_counts):int((train_split+val_split)*min_counts)] + two_indices[int(train_split*min_counts):int((train_split+val_split)*min_counts)] + three_indices[int(train_split*min_counts):int((train_split+val_split)*min_counts)]
            test_indices = zero_indices[int((train_split+val_split)*min_counts):] + one_indices[int((train_split+val_split)*min_counts):] + two_indices[int((train_split+val_split)*min_counts):] + three_indices[int((train_split+val_split)*min_counts):]

            train_channels = [[data[i], i] for i in train_indices]
            val_channels = [[data[i], i] for i in val_indices]
            test_channels = [[data[i], i] for i in test_indices]
        else:
            # Train, val, test split (all indices)
            train_indices = random.sample(range(len(data)), int(train_split*len(data)))
            val_indices = random.sample(range(len(data)), int(val_split*len(data)))
            test_indices = random.sample(range(len(data)), len(data) - int(train_split*len(data)) - int(val_split*len(data)))

            train_channels = [[data[i], i] for i in train_indices]
            val_channels = [[data[i], i] for i in val_indices]
            test_channels = [[data[i], i] for i in test_indices]

        # Normalization
        if scale:
            total_channels = {"train":train_channels, "val":val_channels, "test":test_channels}
            for key, channels in total_channels.items():
                for i in range(len(channels)):
                    scaler = StandardScaler()
                    scaler.fit(channels[i][0][0].reshape(-1, 1))
                    channels[i][0][0] = scaler.transform(channels[i][0][0].reshape(-1, 1)) # (num_timesteps, 1)
        else:
            total_channels = {"train":train_channels, "val":val_channels, "test":test_channels}

        if full_channels:
            if median_seq_len: # Use median length sequence of train_chanels as the window size/context size
                window_size = int(stats.median([train_channel[0][0].shape[0] for train_channel in train_channels]))
                window_size //= 4 # Divide median by 4 (as it is too large for most models)
                if median_seq_only:
                    return window_size

            full_channels = {key:[] for key in total_channels.keys()}
            for key, channels in total_channels.items():
                for channel in channels:
                    x = torch.from_numpy(channel[0][0].T).to(dtype)
                    y = torch.tensor(channel[0][1], dtype=dtype)
                    c = torch.tensor(channel[1], dtype=torch.long)

                    # Downsampling
                    if kernel_stride==-1: # Arg for half the kernel size
                        kernel_stride = kernel_size // 2
                    if pool_type=="avg":
                        x = F.avg_pool1d(x, kernel_size=kernel_size, stride=kernel_stride)
                    elif pool_type=="max":
                        x = F.max_pool1d(x, kernel_size=kernel_size, stride=kernel_stride)

                    full_channels[key].append((x, y, c))

            train_data = [full_channels["train"][i][0] for i in range(len(full_channels["train"]))]
            train_labels = [full_channels["train"][i][1] for i in range(len(full_channels["train"]))]
            train_ch_ids = [full_channels["train"][i][2] for i in range(len(full_channels["train"]))]
            val_data = [full_channels["val"][i][0] for i in range(len(full_channels["val"]))]
            val_labels = [full_channels["val"][i][1] for i in range(len(full_channels["val"]))]
            val_ch_ids = [full_channels["val"][i][2] for i in range(len(full_channels["val"]))]
            test_data = [full_channels["test"][i][0] for i in range(len(full_channels["test"]))]
            test_labels = [full_channels["test"][i][1] for i in range(len(full_channels["test"]))]
            test_ch_ids = [full_channels["test"][i][2] for i in range(len(full_channels["test"]))]

            if resizing_mode in {"pad_trunc", "resize"}:
                # Pad + truncate or resize all sequences to window_size
                train_data = resize_sequence(train_data, window_size, resizing_mode).to(dtype)
                val_data = resize_sequence(val_data, window_size, resizing_mode).to(dtype)
                test_data = resize_sequence(test_data, window_size, resizing_mode).to(dtype)

                # Conver label and ch_ids from lists of numpy arrays to tensors
                train_labels = torch.from_numpy(np.array(train_labels))
                train_ch_ids = torch.from_numpy(np.array(train_ch_ids))
                val_labels = torch.from_numpy(np.array(val_labels))
                val_ch_ids = torch.from_numpy(np.array(val_ch_ids))
                test_labels = torch.from_numpy(np.array(test_labels))
                test_ch_ids = torch.from_numpy(np.array(test_ch_ids))

            return (train_data, train_labels, train_ch_ids, val_data, val_labels, val_ch_ids, test_data, test_labels, test_ch_ids, window_size)

        # Create windows
        total_windows, total_labels, total_ch_ids = [], [], []

        for key, channels in total_channels.items(): # Iterate: Train -> Val -> Test

            if np.isclose(eval(f"{key}_split"), 0, atol=1e-9):
                total_windows.append(torch.empty((0, window_size), dtype=dtype))
                total_labels.append(torch.empty((0), dtype=dtype))
                total_ch_ids.append(torch.empty((0), dtype=torch.long))
                continue

            windows, labels, ch_ids = [], [], []
            for channel in channels: # Iterate over each channel
                x = torch.from_numpy(channel[0][0].T).to(dtype)
                y = torch.tensor(channel[0][1], dtype=dtype)

                # Downsampling
                if kernel_stride==-1:
                    kernel_stride = kernel_size // 2 # Arg for half the kernel size
                if pool_type=="avg":
                    x = F.avg_pool1d(x, kernel_size=kernel_size, stride=kernel_stride)
                elif pool_type=="max":
                    x = F.max_pool1d(x, kernel_size=kernel_size, stride=kernel_stride)

                # Create windows (num_windows, window_size)
                x = create_windows(x.squeeze(0), window_size=window_size, window_stride=window_stride, resizing_mode=resizing_mode)

                # Create labels
                y = torch.tensor([y]*x.shape[0], dtype=dtype)

                # Create Channel IDs
                c = torch.tensor(channel[1], dtype=torch.long)
                c = torch.tensor([c]*x.shape[0], dtype=torch.long)

                # Append windows and labels
                windows.append(x); labels.append(y); ch_ids.append(c);

            # Concatenate all examples
            total_windows.append(torch.cat(windows, dim=0).to(dtype))
            total_labels.append(torch.cat(labels, dim=0).to(dtype))
            total_ch_ids.append(torch.cat(ch_ids, dim=0))

        train_data, val_data, test_data = total_windows
        train_labels, val_labels, test_labels = total_labels
        train_ch_ids, val_ch_ids, test_ch_ids = total_ch_ids

        x = [train_data, train_labels, train_ch_ids, val_data, val_labels, val_ch_ids, test_data, test_labels, test_ch_ids]

        if multicluster:
            x.append(num_channels)

        return tuple(x)

def load_open_neuro_multicluster(patient_clusters,
                                 kernel_size,
                                 kernel_stride,
                                 window_size,
                                 window_stride,
                                 dtype,
                                 pool_type,
                                 balance,
                                 scale,
                                 train_split,
                                 val_split,
                                 seed,
                                 task,
                                 full_channels,
                                 resizing_mode,
                                 median_seq_len):
    """
    Load and concatenate data from multiple patient clusters.

    Args:
        Standard arguments for load_open_neuro_interchannel.
    Return:
        total_data (Tuple[torch.Tensor]): Concatenated data from all patient clusters in order of: train_data, train_labels, train_ch_ids,
                                          val_data, val_labels, val_ch_ids, test_data, test_labels, test_ch_ids
    """

    # If median_seq_len is True, calculate the median sequence length for each cluster. Then average the medians to obtain the window size
    median_window_size=0
    if median_seq_len:
        print(f"Calculating median sequence length...")
        # Iterate through all channels in all training clusters and find their sequence lengths
        for patient_cluster in patient_clusters:
            median_window_size += load_open_neuro_interchannel(
                patient_cluster=patient_cluster,
                kernel_size=kernel_size,
                kernel_stride=kernel_stride,
                window_size=window_size,
                dtype=dtype,
                window_stride=window_stride,
                pool_type=pool_type,
                balance=balance,
                scale=scale,
                train_split=train_split,
                val_split=val_split,
                seed=seed,
                task=task,
                multicluster=True,
                full_channels=full_channels,
                resizing_mode=resizing_mode,
                median_seq_len=median_seq_len,
                median_seq_only=True,
            )
        median_window_size //= len(patient_clusters)
        print(f"Median: {median_window_size}")


    num_channels = 0
    data_lists = [[] for _ in range(9)]
    for i, patient_cluster in enumerate(patient_clusters):
        cluster_data = load_open_neuro_interchannel(
            patient_cluster=patient_cluster,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            window_size=median_window_size if median_seq_len else window_size,
            dtype=dtype,
            window_stride=window_stride,
            pool_type=pool_type,
            balance=balance,
            scale=scale,
            train_split=train_split,
            val_split=val_split,
            seed=seed,
            task=task,
            multicluster=True,
            full_channels=full_channels,
            resizing_mode=resizing_mode,
            median_seq_len=False,
        )
        num_channels += cluster_data[-1] if not full_channels else 0

        # Append cluster data
        for j in range(9):
            if i!=0 and j in {2, 5, 8}: # Create Unique Channel IDs
                appended_data = cluster_data[j] + num_channels if not full_channels else cluster_data[j]
                data_lists[j].append(appended_data)
            else:
                data_lists[j].append(cluster_data[j])

    # Check that all Channel IDs are indeed unique
    if not full_channels:
        for j in {2, 5, 8}:
            all_ch_ids = set()
            for ch_ids_tensor in data_lists[j]:
                unique_ch_ids = set(torch.unique(ch_ids_tensor).tolist())
                if not unique_ch_ids.isdisjoint(all_ch_ids): # Check if the intersection with 'all_ch_ids' is empty
                    raise ValueError (f"Common elements found for index {j}")

    # Concatenate all clusters train, val, and test data together
    if full_channels and resizing_mode not in {"pad_trunc", "resizing"}:
        total_data = [list(chain(*data_lists[i])) for i in range(9)]
    else:
        total_data = [torch.cat(data_lists[i], dim=0) for i in range(9)]

    if median_seq_len:
        total_data.append(median_window_size)

    return tuple(total_data)

def load_open_neuro_loocv(train_clusters,
                          test_clusters,
                          kernel_size,
                          kernel_stride,
                          window_size,
                          window_stride,
                          dtype,
                          pool_type,
                          balance,
                          scale,
                          train_split,
                          val_split,
                          seed,
                          task,
                          loader_type="train",
                          full_channels=False,
                          resizing_mode="None",
                          median_seq_len=False):

    if loader_type in {"train", "all"}:
        total_train_data = load_open_neuro_multicluster(train_clusters,
                                            kernel_size,
                                            kernel_stride,
                                            window_size,
                                            window_stride,
                                            dtype,
                                            pool_type,
                                            balance,
                                            scale,
                                            train_split,
                                            val_split,
                                            seed,
                                            task,
                                            full_channels,
                                            resizing_mode,
                                            median_seq_len)

        # Get training clusters' data
        train_data, train_labels, train_ch_ids, val_data, val_labels, val_ch_ids, test_data, test_labels, test_ch_ids = total_train_data[:9]

        if median_seq_len:
            window_size = total_train_data[-1]

        # Combine test data into training data. Use ALL data from the cluster(s) for training
        if full_channels and resizing_mode not in {"pad_trunc", "resizing"}:
            train_data = train_data + test_data
            train_labels = train_labels + test_labels
            train_ch_ids = train_ch_ids + test_ch_ids
        else:
            train_data = torch.cat([train_data, test_data], dim=0)
            train_labels = torch.cat([train_labels, test_labels], dim=0)
            train_ch_ids = torch.cat([train_ch_ids, test_ch_ids], dim=0)

        del total_train_data
        gc.collect()

        train_returns = [train_data, train_labels, train_ch_ids, val_data, val_labels, val_ch_ids]

        if median_seq_len:
            train_returns.append(window_size)

        if loader_type=="train":
            return tuple(train_returns)

    if loader_type in {"test", "all"}:
        total_test_data = load_open_neuro_multicluster(test_clusters,
                                        kernel_size,
                                        kernel_stride,
                                        window_size,
                                        window_stride,
                                        dtype,
                                        pool_type,
                                        balance,
                                        scale,
                                        train_split, # train_split = 0
                                        val_split, # val_split = 0 => only test data
                                        seed,
                                        task,
                                        full_channels,
                                        resizing_mode,
                                        median_seq_len=False)

        # Get last 3 entries
        data1, labels1, ch_ids1, data2, labels2, ch_ids2, data3, labels3, ch_ids3 = total_test_data

        if full_channels and resizing_mode not in {"pad_trunc", "resizing"}:
            test_data = data1 + data2 + data3
            test_labels = labels1 + labels2 + labels3
            test_ch_ids = ch_ids1 + ch_ids2 + ch_ids3
        else:
            test_data = torch.cat([data1, data2, data3], dim=0)
            test_labels = torch.cat([labels1, labels2, labels3], dim=0)
            test_ch_ids = torch.cat([ch_ids1, ch_ids2, ch_ids3], dim=0)

        del total_test_data
        gc.collect()

        if loader_type=="test":
            return (test_data, test_labels, test_ch_ids)

        if median_seq_len:
            train_returns.pop()
            test_returns = [test_data, test_labels, test_ch_ids, window_size]
        else:
            test_returns = [test_data, test_labels, test_ch_ids]

        if loader_type=="all":
            return tuple(train_returns + test_returns)
        else:
            raise ValueError(f"Invalid loader_type {loader_type}. Must be 'train', 'test', or 'all'")



def get_rocket_features(data, num_kernels, max_dilation, seed):
    """
    Create a Rocket transform object.

    Args:
        data (List[np.ndarray]): List of time series data each are 1D numpy arrays of shape (*,).
        num_kernels (int): Number of kernels to use.
        max_dilation (int): Maximum dilation to use for the kernels.
        seed (int): Random seed for reproducibility.
    Returns:
        transformed (torch.Tensor): A tensor of transformed data, of shape (n, 2*num_kernels) where n is the number of time series.
    """

    # Generate kernels
    kernels = generate_kernels(max_dilation, num_kernels, seed)

    # Apply kernels
    n = len(data)
    num_features = 2*num_kernels
    rocket_features = torch.zeros(n, num_features)

    for i in range(n):
        input = data[i].double().cpu().numpy()
        kernel_out = apply_kernels(input, kernels)
        rocket_features[i] = torch.from_numpy(kernel_out).squeeze()

    return rocket_features


#<-----------------UEA Classification------------------------>
def uea(dataset_name, mode="univariate"):
    """
    Load a UEA dataset for classification. For more information see: https://www.timeseriesclassification.com/

    Args:
        dataset_name (str): Name of the dataset corresponding to either one of the univariate or multivariate UEA datasets.\
        mode (str): Type of time series data. Options are "univariate" or "multivariate".
    """

    # Load raw data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))

    train_data, train_labels = load_from_tsfile(os.path.join(root_dir, f"data/classification/uea/{mode}/{dataset_name}/{dataset_name}_TRAIN.ts"))
    test_data, test_labels = load_from_tsfile(os.path.join(root_dir, f"data/classification/uea/{mode}/{dataset_name}/{dataset_name}_TEST.ts"))

    # TODO: Add separate preprocessing if time series is variable length.
    # TODO: Add window creation mode for fixed lengths.

    return (train_data, train_labels, test_data, test_labels)


#<-----------------General Usage------------------------>

def get_loader(args,
               rank,
               world_size,
               data,
               labels=None,
               ch_ids=None,
               flag="sl",
               shuffle=False,
               generator=torch.Generator()):
    """
    Returns a DataLoader for a specific task.


    Returns:
        DataLoader (Optional): A PyTorch DataLoader object for the a dataset class.
        Dataset (Optional): A PyTorch Dataset object for the a dataset class.
    """
    dataset_class = eval(f"args.{flag}.dataset_class")

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    if args.data.rocket_transform:
        data = get_rocket_features(data, args.sl.num_kernels, args.sl.max_dilation, args.exp.seed)

    if dataset_class == "forecasting":
        dataset = ForecastingDataset(data, args.data.seq_len, args.data.pred_len, args.data.dtype)
    elif dataset_class == "univariate_forecasting":
        dataset = UnivariateForecastingDataset(data[0], data[1], args.data.dtype)
    elif dataset_class == "classification":
        dataset = ClassificationDataset(data, labels, ch_ids, args.open_neuro.task, args.data.full_channels)
    elif dataset_class == "variable_length":
        dataset = VariableLengthDataset(data, labels, args.data.pad_to_max, eval(f"args.{flag}.batch_size"), args.data.tslearn, args.data.numpy_data, args.data.dtype)
    else:
        raise ValueError(f"Invalid dataset class: {dataset_class}")

    if args.data.dataset_only:
        return dataset

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=args.exp.seed, shuffle=args.ddp.shuffle) if args.ddp.ddp else None
    shuffle = False if args.ddp.ddp else shuffle
    return DataLoader(dataset,
                      batch_size=eval(f"args.{flag}.batch_size"),
                      shuffle=shuffle,
                      drop_last=args.data.drop_last,
                      num_workers=args.data.num_workers,
                      generator=generator if not args.ddp.ddp else None,  # Only pass generator if not using ddp (seed from distributed sampler is used instead)
                      sampler=sampler,
                      persistent_workers=True,
                      prefetch_factor=args.data.prefetch_factor,
                      pin_memory=args.data.pin_memory)

def get_loaders(args, flag="sl", generator=torch.Generator(), rank=0, world_size=1, dataset_class="forecasting", loader_type="train", dataset_only=False):
    if args.data.dataset in {"ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "traffic", "weather", "illness", "rf_emf"}:
        train_data, val_data, test_data = load_forecasting(dataset_name=args.data.dataset,
                                                           seq_len=args.data.seq_len,
                                                           pred_len=args.data.pred_len,
                                                           window_stride=args.data.window_stride,
                                                           scale=args.data.scale,
                                                           train_split=args.data.train_split,
                                                           val_split=args.data.val_split,
                                                           univariate=args.data.univariate,
                                                           resizing_mode=args.data.resizing_mode,
                                                           target_channel=args.data.target_channel,)
        train_labels = val_labels = test_labels = train_ch_ids = val_ch_ids = test_ch_ids = None
    elif args.data.dataset=="open_neuro":
        if args.open_neuro.all_clusters:
            if args.open_neuro.task=="binary":
                patient_clusters={"jh", "pt", "ummc", "umf"}
            elif args.open_neuro.task=="multi":
                patient_clusters={"pt", "ummc"}
            x = load_open_neuro_multicluster(
                patient_clusters=patient_clusters,
                kernel_size=args.open_neuro.kernel_size,
                kernel_stride=args.open_neuro.kernel_stride,
                window_size=args.data.seq_len,
                window_stride=args.data.window_stride,
                dtype=args.data.dtype,
                pool_type=args.open_neuro.pool_type,
                balance=args.data.balance,
                scale=args.data.scale,
                train_split=args.data.train_split,
                val_split=args.data.val_split,
                seed=args.exp.seed,
                task=args.open_neuro.task,
                full_channels=args.data.full_channels,
                resizing_mode=args.data.resizing_mode,
                median_seq_len=args.data.median_seq_len
            )
            train_data, train_labels, train_ch_ids, val_data, val_labels, val_ch_ids, test_data, test_labels, test_ch_ids = x[:9]

        elif args.open_neuro.loocv:
            x = load_open_neuro_loocv(
                train_clusters=args.open_neuro.train_clusters,
                test_clusters=args.open_neuro.test_clusters,
                kernel_size=args.open_neuro.kernel_size,
                kernel_stride=args.open_neuro.kernel_stride,
                window_size=args.data.seq_len,
                window_stride=args.data.window_stride,
                dtype=args.data.dtype,
                pool_type=args.open_neuro.pool_type,
                balance=args.data.balance,
                scale=args.data.scale,
                train_split=args.data.train_split,
                val_split=args.data.val_split,
                seed=args.exp.seed,
                task=args.open_neuro.task,
                loader_type=loader_type,
                full_channels=args.data.full_channels,
                resizing_mode=args.data.resizing_mode,
                median_seq_len=args.data.median_seq_len
            )

            if loader_type=="train":
                train_data, train_labels, train_ch_ids, val_data, val_labels, val_ch_ids = x[:6]
            elif loader_type=="test":
                test_data, test_labels, test_ch_ids = x
            elif loader_type=="all":
                train_data, train_labels, train_ch_ids, val_data, val_labels, val_ch_ids, test_data, test_labels, test_ch_ids = x[:9]


        else:
            x = load_open_neuro_interchannel(
                patient_cluster=args.open_neuro.patient_cluster,
                kernel_size=args.open_neuro.kernel_size,
                kernel_stride=args.open_neuro.kernel_stride,
                window_size=args.data.seq_len,
                window_stride=args.data.window_stride,
                dtype=args.data.dtype,
                pool_type=args.open_neuro.pool_type,
                balance=args.data.balance,
                scale=args.data.scale,
                train_split=args.data.train_split,
                val_split=args.data.val_split,
                seed=args.exp.seed,
                task=args.open_neuro.task,
                multicluster=False,
                full_channels=args.data.full_channels,
                resizing_mode=args.data.resizing_mode,
                median_seq_len=args.data.median_seq_len
            )
            train_data, train_labels, train_ch_ids, val_data, val_labels, val_ch_ids, test_data, test_labels, test_ch_ids = x[:9]
    else:
        raise ValueError(f"Invalid dataset name: {args.data.dataset}")

    if loader_type in {"train", "all"}:
        train_loader = get_loader(args=args,
                                data=train_data,
                                labels=train_labels,
                                shuffle=True,
                                rank=rank,
                                world_size=world_size,
                                generator=generator,
                                ch_ids=train_ch_ids,
                                flag=flag)
        val_loader = get_loader(args=args,
                                data=val_data,
                                labels=val_labels,
                                shuffle=True,
                                rank=rank,
                                world_size=world_size,
                                generator=generator,
                                ch_ids=val_ch_ids,
                                flag=flag)
    if loader_type in {"test", "all"}:
        test_loader = get_loader(args=args,
                                data=test_data,
                                labels=test_labels,
                                shuffle=args.data.shuffle_test,
                                rank=rank,
                                world_size=world_size,
                                generator=generator,
                                ch_ids=test_ch_ids,
                                flag=flag)

    loaders = []

    if loader_type=="train":
        loaders.append(train_loader); loaders.append(val_loader)
    elif loader_type=="test":
        loaders.append(test_loader)
    elif loader_type=="all":
        loaders.append(train_loader); loaders.append(val_loader); loaders.append(test_loader)
    else:
        raise ValueError(f"Invalid loader type: {loader_type}")

    if args.data.median_seq_len:
        window_size = x[-1]
        loaders.append(window_size)

    return tuple(loaders)

# Test
if __name__=="__main__":

    x = load_open_neuro_interchannel(patient_cluster="umf",
                            kernel_size=150,
                            kernel_stride=75,
                            window_size=512,
                            window_stride=24,
                            pool_type="avg",
                            balance=True,
                            scale=True,
                            train_split=0.6,
                            val_split=0.2,
                            seed=1995,
                            task="binary",
                            full_channels=False,
                            multicluster=False,
                            resizing_mode="None")

    train_data, train_labels, train_ch_ids, val_data, val_labels, val_ch_ids, test_data, test_labels, test_ch_ids = x

    print(f"Train Data: {train_data.shape}, Train Labels: {train_labels.shape}, Train Ch IDs: {train_ch_ids.shape}")
    print(f"Val Data: {val_data.shape}, Val Labels: {val_labels.shape}, Val Ch IDs: {val_ch_ids.shape}")
    print(f"Test Data: {test_data.shape}, Test Labels: {test_labels.shape}, Test Ch IDs: {test_ch_ids.shape}")
