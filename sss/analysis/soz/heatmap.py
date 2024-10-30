import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sss.utils.dataloading import load_open_neuro_interchannel, create_windows
import random
from torch.utils.data import TensorDataset, DataLoader
from sss.utils.recalibrate import train_calibrator, load
from rich.console import Console
import gc
import torch.nn.functional as F

def heatmap_analysis(
    run,
    run_id,
    model,
    device,
    new_seed=False,
    mode="train",
    window_stride=1,
    batch_size=4096,
    num_sampled_channels=1,
    patient_cluster="umf",
    avg_mode="none",
    save=True,
    method_name="SSS",
    cmap='viridis',
    dpi=300,
    calibrate=False,
    calibration_model="isotonic_regression",
    relative=False,
    probability=True,
):
    # Rich console
    console = Console()

    # Set up the seed
    seed = random.randint(0, 50000) if new_seed else run["parameters/exp/seed"].fetch()

    # Load data parameters
    window_size = run["parameters/data/seq_len"].fetch()

    if calibrate:
        with torch.no_grad():
            calibrator = get_calibrator(device, run_id, calibration_model, console)

    # Load the data
    full_channels = load_open_neuro_interchannel(
        patient_cluster=patient_cluster,
        kernel_size=run["parameters/open_neuro/kernel_size"].fetch(),
        kernel_stride=run["parameters/open_neuro/kernel_stride"].fetch(),
        window_size=window_size,
        window_stride=run["parameters/data/window_stride"].fetch(),
        dtype=run["parameters/data/dtype"].fetch(),
        pool_type=run["parameters/open_neuro/pool_type"].fetch(),
        balance=run["parameters/data/balance"].fetch(),
        scale=run["parameters/data/scale"].fetch(),
        train_split=run["parameters/data/train_split"].fetch(),
        val_split=run["parameters/data/val_split"].fetch(),
        seed=seed,
        task=run["parameters/open_neuro/task"].fetch(),
        full_channels=True,
        multicluster=False,
        resizing_mode="None",
        median_seq_len=False
    )
    train_data, train_labels, train_ch_ids, val_data, val_labels, val_ch_ids, test_data, test_labels, test_ch_ids, _ = full_channels

    # Select the appropriate dataset based on the mode
    mode_mapping = {
        "train": (train_data, train_labels, train_ch_ids),
        "val": (val_data, val_labels, val_ch_ids),
        "test": (test_data, test_labels, test_ch_ids)
    }
    data, labels, ch_ids = mode_mapping[mode]

    num_channels = len(data)
    for i in range(min(num_sampled_channels, num_channels)):
        # Get channel data
        channel = data[i].squeeze()
        channel_length = len(channel)

        if window_size > channel_length:
            console.log(f"Skipping channel {i}: window size ({window_size}) > channel length ({channel_length})")
            continue

        console.log(f"Processing channel {i}: Window size: {window_size}, Channel length: {channel_length}")

        # Create windows
        windows = create_windows(channel, window_size, window_stride=1)
        windows_set = TensorDataset(windows)
        windows_loader = DataLoader(windows_set, batch_size=batch_size, shuffle=False)

        # Get window probabilities
        window_probs = []
        with torch.no_grad():
            for batch in windows_loader:
                inputs = batch[0].to(device)
                pred = model(inputs)
                window_probs.append(pred.cpu().numpy())

        window_probs_arr = np.concatenate(window_probs, axis=0).squeeze()

        # Apply sigmoid function
        if probability:
            window_probs_arr = F.sigmoid(torch.tensor(window_probs_arr)).numpy()
            console.log(f"Applied sigmoid function to window probabilities for channel {i}")
            console.log(f"Max probability: {np.max(window_probs_arr)}")
            console.log(f"Min probability: {np.min(window_probs_arr)}")

        if calibrate:
            window_probs_arr = calibrator.calibrate(window_probs_arr).cpu().numpy()
            console.log(f"Calibrated window probabilities for channel {i} with {calibration_model}")

        # Distribute and average probabilities
        avg_probs = np.zeros(channel_length)
        counts = np.zeros(channel_length)
        for j, prob in enumerate(window_probs_arr):
            start_idx = j
            end_idx = start_idx + window_size
            avg_probs[start_idx:end_idx] += prob
            counts[start_idx:end_idx] += 1
        avg_probs = np.divide(avg_probs, counts, where=counts!=0)

        # Apply moving average if specified
        if avg_mode == "sma":
            avg_probs = simple_moving_average(avg_probs, window_size)
        elif avg_mode == "ema":
            avg_probs = exp_moving_average(avg_probs, window_size)
        elif avg_mode == "local_var":
            avg_probs = local_variance(avg_probs, window_size)
        elif avg_mode == "none":
            pass
        else:
            raise ValueError(f"Invalid avg_mode: {avg_mode}. Options: 'sma', 'ema', 'local_var', 'none'")

        # Plot the heatmap
        label = labels[i].item()
        ch_id = ch_ids[i].item()
        plot_heatmap(
            avg_probs,
            channel.cpu().numpy(),
            label,
            ch_id,
            run_id,
            patient_cluster,
            method_name,
            cmap,
            save,
            dpi,
            avg_mode,
            console,
            relative
        )

    console.log(f"Finished processing {min(num_sampled_channels, num_channels)} channels.")

def simple_moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def exp_moving_average(x, w):
    alpha = 2 /(w + 1.0)
    alpha_rev = 1-alpha
    n = x.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = x[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = x*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

def local_variance(x, w):
    """
    Calculates the variance of probabilities within a sliding window.
    High variance regions might indicate areas of uncertainty or transition.
    """
    variance = np.zeros(len(x))
    padded_probs = np.pad(x, (w//2, w//2), mode='edge')

    for i in range(len(x)):
        window = padded_probs[i:i+w]
        variance[i] = np.var(window)

    return variance


def get_calibrator(device, run_id, calibration_model, console):
    model, loaders, calibrator, exp_args = load(device, run_id, calibration_model, console)
    train_calibrator(exp_args, loaders, calibrator, model, device)
    return calibrator


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# Set up the font to mimic LaTeX/Computer Modern
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False  # Ensures proper minus sign


def plot_heatmap(avg_probs, channel, label, ch_id=14, run_id="SOZ-33", patient_cluster="jh", method_name="SSS", cmap='viridis', save=False, dpi=300, avg_mode="sma", console=None, relative=False):
    assert cmap in ['viridis', 'plasma', "inferno", "coolwarm"], "Invalid color map. Options: 'viridis', 'plasma', 'inferno', 'coolwarm'."

    label_name = "SOZ" if label==1 else "Non-SOZ"
    cluster_name = "All Clusters" if patient_cluster == "all" else f"Cluster {patient_cluster}"
    num_time_steps = len(channel)

    heatmap_height = 200
    heatmap_data = np.tile(avg_probs, (heatmap_height, 1))

    fig, ax = plt.subplots(figsize=(16, 8), dpi=600)
    plt.subplots_adjust(left=0.12, right=0.9, top=0.9, bottom=0.1)

    X, Y = np.meshgrid(np.arange(num_time_steps), np.linspace(np.min(channel), np.max(channel), heatmap_height))

    console.log(f"Max probability: {np.max(avg_probs)}")
    console.log(f"Min probability: {np.min(avg_probs)}")

    if relative:
        vmin = np.min(avg_probs)
        vmax = np.max(avg_probs)
        cbar_label = "Relative Probability of SOZ"
    else:
        vmin = 0
        vmax = 1
        cbar_label = "Probability of SOZ"

    pcm = ax.pcolormesh(X, Y, heatmap_data, shading='auto', cmap=cmap, alpha=0.7, vmin=vmin, vmax=vmax)
    ax.plot(channel, color='black', linewidth=1.5, alpha=0.85)

    cluster_mapping = {
        "jh": "JHH",
        "pt": "NIH",
        "ummc": "UMMC",
        "umf": "UMH",
    }

    # Reduced title font size by ~25%
    # ax.set_title(f'{label_name} iEEG Signal ({cluster_mapping[patient_cluster]})', fontsize=28, pad=10)
    # Updated title with full LaTeX rendering and inline math for B=128
    ax.set_title(f'{label_name} iEEG Signal ({cluster_mapping[patient_cluster]}) $L=1024$', fontsize=28, pad=10)

    # Adjusted x-axis and y-axis label font sizes
    ax.set_xlabel('Time', fontsize=24)
    ax.set_ylabel('Normalized Amplitude', fontsize=24)

    divider = make_axes_locatable(ax)

    # Colorbar
    cax = divider.append_axes("right", size="2.5%", pad=0.08)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.set_label(cbar_label, fontsize=22, labelpad=10)
    cbar.ax.tick_params(labelsize=14)

    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=18)

    if save:
        save_dir = f"../../../figures/{run_id}_{patient_cluster}_{avg_mode}"
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{run_id}_channel_{ch_id}_{label_name}_{patient_cluster}.png")
        console.log(f"Saving figure with DPI {dpi}...")
        plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
