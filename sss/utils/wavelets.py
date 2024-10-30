import numpy as np
import torch
import pywt
import ptwt
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from typing import Tuple, List

from sss.utils.dataloading import load_open_neuro_interchannel

def wavelet_features(
    signal: torch.Tensor,
    fs: int,
    num_scales:int = 35,
    f_min:float = 0.5,
    f_max:float = 300.0,
    wavelet:str = 'cmor1.5-1.0') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the continuous wavelet transform to a signal and return coefficients and frequencies.
    Optimized for intracranial EEG (iEEG) recordings of seizure activity to capture the ranges of:
        - Delta (0.5-4 Hz)
        - Theta (4-8 Hz)
        - Alpha (8-13 Hz)
        - Beta (13-30 Hz)
        - Gamma (30-100 Hz)
        - High-Frequency Oscillations (HFOs, 80-500 Hz)

    Nyquist Frequency: It's important to note that the maximum frequency we can reliably analyze is half the sampling frequency,
    known as the Nyquist frequency. For example, given a signal with 1000 Hz sampling rate, this would be 500 Hz.

    Suggested Frequency Range [f_min, f_max]. We recommend utilizing a range from f_min=0.5 Hz up to f_max=250-300 Hz,
    for a signal with sampling frequency fs=1000Hz, for example. This covers all the main frequency bands of interest
    in epilepsy research, including:
        - Traditional lower frequency bands (delta, theta, alpha, beta)
        - Gamma oscillations
        - A significant portion of the HFO range

    Number of Scales. Something like 30-40 scales could provide a good balance between resolution and computational efficiency,
    for a signal with fs=1000Hz. Fewer scales may not capture the full range of frequencies, while more scales will be computationally expensive.

    Logarithmic vs Linear Spacing: Logarithmic spacing of frequencies is often preferred in this context as it provides better resolution
    in the lower frequency bands while still capturing higher frequency activity.

    Args:
        signal (torch.Tensor): The input iEEG time series of shape (batch_size, signal_length).
        fs (float): Sampling frequency of the signal in Hz.
        num_scales (int, optional): Number of wavelet scales to use. Defaults to 35.
        f_min (float, optional): Minimum frequency to analyze in Hz. Defaults to 0.5 Hz.
        f_max (float, optional): Maximum frequency to analyze in Hz. Defaults to 300 Hz.
        wavelet (str, optional): Wavelet to use. Defaults to 'morl' (Morlet wavelet). Options:

    Returns:
        tuple: A tuple containing:
            - coefficients (torch.Tensor): 3D tensor of wavelet coefficients of shape (batch_size, num_scales, signal_length)
            - frequencies (torch.Tensor): 1D tensor of corresponding frequencies.
    """
    with torch.no_grad():
        # Ensure signal is a PyTorch tensor
        if not isinstance(signal, torch.Tensor):
            signal = torch.from_numpy(signal)

        # Ensure signal is 1D
        if signal.dim() > 1:
            signal = signal.squeeze()

        # Ensure f_max doesn't exceed Nyquist frequency
        f_max = min(f_max, fs / 2)

        # Calculate frequencies using logarithmic spacing
        frequencies = np.logspace(np.log10(f_min), np.log10(f_max), num_scales)

        # Normalize frequencies by sampling frequency
        normalized_frequencies = frequencies / fs

        # Convert normalized frequencies to scales
        scales = pywt.frequency2scale(wavelet, normalized_frequencies)

        # Convert NumPy array to PyTorch tensor
        scales = torch.tensor(scales)

        # Sort scales in descending order
        scales, idx = torch.sort(scales, descending=True)
        frequencies = torch.tensor(frequencies)[idx]  # Reorder frequencies to match scales

        # Perform continuous wavelet transform using ptwt
        wavelet_object = pywt.ContinuousWavelet(wavelet)

        try:
            coefficients, _ = ptwt.continuous_transform.cwt(signal.unsqueeze(0), scales, wavelet_object)
            coefficients = coefficients.squeeze(0)  # Remove batch dimension
        except ValueError as e:
            print(f"Error in cwt: {e}")
            print(f"Min scale: {scales.min().item()}, Max scale: {scales.max().item()}")
            print(f"Min frequency: {frequencies.min().item()}, Max frequency: {frequencies.max().item()}")
            raise

    return coefficients.permute(1, 0, 2), frequencies

def plot_signal_and_scalogram(time, signal, coefficients, frequencies, title="Signal and Wavelet Transform", save_scalogram=False, interpolate=False):

    signal = signal.squeeze().numpy()
    coefficients = coefficients.squeeze().numpy()
    frequencies = frequencies.squeeze().numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot original signal
    ax1.plot(time, signal)
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Original Signal')

    # Prepare scalogram data
    if interpolate:
        t = np.linspace(time[0], time[-1], 1000)
        f = np.linspace(frequencies[-1], frequencies[0], 1000)
        interp_func = interp2d(time, frequencies, np.abs(coefficients), kind='cubic')
        scalogram_data = interp_func(t, f)
    else:
        scalogram_data = np.abs(coefficients)

    im = ax2.imshow(scalogram_data, extent=[time[0], time[-1], frequencies[-1], frequencies[0]],
                    aspect='auto', interpolation='nearest' if not interpolate else 'bilinear', cmap='jet')
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Time')
    ax2.set_title('Wavelet Transform Scalogram')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('Magnitude')

    plt.suptitle(title)
    plt.tight_layout()

    # Save the scalogram if requested
    if save_scalogram:
        save_path = os.path.join(os.getcwd(), "scalogram.png")
        fig_save, ax_save = plt.subplots(figsize=(15, 6))

        # Plot the scalogram
        im_save = ax_save.imshow(scalogram_data,
                                 extent=[time[0], time[-1], frequencies[-1], frequencies[0]],
                                 aspect='auto',
                                 interpolation='nearest' if not interpolate else 'bilinear',
                                 cmap='jet')

        # Remove all axes, ticks, and labels
        ax_save.axis('off')

        # Remove any extra white space around the image
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)

        # Save the figure without any extra padding
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig_save)
        print(f"Scalogram saved as: {save_path}")

    plt.show()


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load the data
    data = load_open_neuro_interchannel(patient_cluster="jh",
                            kernel_size=24,
                            kernel_stride=12,
                            window_size=1024,
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
                            median_seq_only=False)

    # Convert data to numpy and apply wavelet transform
    import random
    import secrets
    import numpy as np


    i = 40012
    signal = data[0][i]  # Assuming this is a 1D signal
    label = data[1][i].item()
    time = np.arange(len(signal))
    fs = 1000  # Sample rate of 1000 Hz
    num_scales = 35
    f_min = 0.5
    f_max = 300
    wavelet = 'morl'
    transformed_coeffs, frequencies = wavelet_features(signal, fs, num_scales, f_min, f_max, wavelet)

    label_name = "SOZ" if label == 1 else "Non-SOZ"
    print(f"{label_name} signal")
    print(f"Original data shape: {signal.shape}")
    print(f"Transformed coefficients shape: {transformed_coeffs.shape}")
    print(f"Frequencies shape: {frequencies.shape}")

    # Plot the signal and scalogram
    plot_signal_and_scalogram(time, signal, transformed_coeffs, frequencies)
