import numpy as np
import torch
import torch.nn as nn
import pywt
import matplotlib.pyplot as plt

class WaveletTimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def wavelet_transform_series(data, wavelet='db4', level=1):
    """Apply wavelet transform to each feature independently."""
    transformed = []
    for i in range(data.shape[1]):
        coeffs = pywt.wavedec(data[:, i], wavelet, level=level)
        transformed.append(np.concatenate(coeffs))
    return np.array(transformed).T

def inverse_wavelet_transform_series(data, original_shape, wavelet='db4', level=1):
    """Apply inverse wavelet transform to each feature independently."""
    reconstructed = np.zeros(original_shape)
    for i in range(original_shape[1]):
        # Calculate lengths of coefficients
        coeffs_len = [len(pywt.wavedec(np.zeros(original_shape[0]), wavelet, level=level)[j]) for j in range(level+1)]
        # Split the data back into coefficient arrays
        coeffs = np.split(data[:, i], np.cumsum(coeffs_len)[:-1])
        reconstructed[:, i] = pywt.waverec(coeffs, wavelet)
    return reconstructed

import numpy as np

def modify_wavelet_coeffs(data, modification_factor=0.3, additional_noise_factor=0.5):
    # Original Gaussian noise
    base_noise = np.random.normal(0, np.std(data, axis=0) * modification_factor, data.shape)

    # Additional noise
    extra_noise = np.random.normal(0, np.std(data, axis=0) * additional_noise_factor, data.shape)

    # Combine noises
    total_noise = base_noise + extra_noise

    return data + total_noise

def wavelet_predictions(model, window, num_samples=100, wavelet='db4', level=1, modification_factor=0.05):
    model.eval()
    original_data = window.squeeze().numpy()

    with torch.no_grad():
        predictions = []
        modified_samples = []
        for _ in range(num_samples):
            # Transform
            transformed_data = wavelet_transform_series(original_data, wavelet, level)
            # Modify
            modified_data = modify_wavelet_coeffs(transformed_data, modification_factor)
            # Inverse transform
            reconstructed_data = inverse_wavelet_transform_series(modified_data, original_data.shape, wavelet, level)
            modified_samples.append(reconstructed_data)

            modified_window = torch.FloatTensor(reconstructed_data).unsqueeze(0)
            output = model(modified_window).squeeze().numpy()
            predictions.append(output)

    return np.array(predictions), np.array(modified_samples)

def visualize_modifications(original_data, modified_samples):
    plt.figure(figsize=(12, 6))
    plt.plot(original_data[:, 0], label='Original', color='black', linewidth=2)
    for i in range(min(5, len(modified_samples))):
        plt.plot(modified_samples[i][:, 0], label=f'Modified {i+1}', alpha=0.5)
    plt.title('Original vs Modified Time Series (First Feature)')
    plt.legend()
    plt.show()

# Example usage
input_size = 1024
hidden_size = 128
output_size = 1
model = WaveletTimeSeriesModel(input_size, hidden_size, output_size)

# Simulating a single window of data
window = torch.randn(1, 100, input_size)  # 1 window with 100 time steps and 5 features

predictions, modified_samples = wavelet_predictions(model, window)

print("Predictions shape:", predictions.shape)
print("Sample predictions:", predictions[:3])

# Visualize the original data vs. modified samples
visualize_modifications(window.squeeze().numpy(), modified_samples)

# Calculate uncertainty
uncertainty = np.var(predictions, axis=0)
print("Uncertainty:", uncertainty)

# Optional: Convert uncertainty to confidence score
confidence_score = 1 / (uncertainty + 1e-6)
print("Confidence score:", confidence_score)
