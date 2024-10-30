import numpy as np
import pywt
import matplotlib.pyplot as plt

def perform_cwt(signal, freq_range, wavelet='cmor1.5-1.0', sampling_rate=1000):
    scales = []
    if wavelet.startswith('cmor'):
        center_frequency = float(wavelet.split('-')[0].replace('cmor', ''))
        scales = [(center_frequency * sampling_rate) / (2 * f * np.pi) for f in freq_range]
    else:
        scales = [1 / (f / sampling_rate) for f in freq_range]

    coeffs, freqs = pywt.cwt(signal, scales, wavelet)

    return coeffs, freqs, scales

# Generate sample EEG with HFO
sampling_rate = 250  # Hz
duration = 4  # second
n_points = sampling_rate * duration
t = np.linspace(0, duration, n_points, endpoint=False)
t_ms = t * 1000  # Convert to milliseconds
signal = np.sin(2 * np.pi * 7 * t) + np.random.normal(0, 0.5, n_points)

# Define frequency range for HFO detection
freq_range = np.linspace(80, 500, 30)

# List of wavelets to try
wavelets = ['cmor1.5-1.0', 'morl', 'mexh', 'gaus1', 'cgau1']

# Plot original signal
plt.figure(figsize=(12, 4))
plt.plot(t_ms, signal)
plt.title('Original EEG Signal')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.show()

for wavelet in wavelets:
    coeffs, freqs, scales = perform_cwt(signal, freq_range, wavelet, sampling_rate)

    print(f"\nWavelet: {wavelet}")
    print(f"Number of scales: {len(scales)}")
    print(f"Shape of coeffs: {coeffs.shape}")

    # Plot CWT results with original signal
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 3]}, sharex=True)

    # Plot original signal
    ax1.plot(t_ms, signal)
    ax1.set_title('Original EEG Signal')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim(t_ms[0], t_ms[-1])

    # Plot CWT
    im = ax2.imshow(np.abs(coeffs), extent=[t_ms[0], t_ms[-1], freqs[-1], freqs[0]], aspect='auto', cmap='jet')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_title(f'CWT of EEG with HFO using {wavelet} wavelet')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('Magnitude')

    plt.tight_layout()
    plt.show()
