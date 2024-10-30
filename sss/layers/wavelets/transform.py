# V1
import torch
import numpy as np
import pywt
import ptwt  # use "from src import ptwt" for a cloned the repo

# Generate dummy time series by adding noise to a sine wave
t = np.linspace(0, 1, 100)
data = np.sin(2 * np.pi * 7 * t) + np.random.normal(0, 0.5, 100)
data_torch = torch.from_numpy(data)
wavelet = pywt.Wavelet('haar')

# compare the forward fwt coefficients
pywt_out = pywt.wavedec(data, wavelet, mode='zero', level=2)
ptwt_out = ptwt.wavedec(data_torch, wavelet, mode='zero', level=2)

# Plot the signal
import matplotlib.pyplot as plt
plt.plot(data)
plt.title('Original Signal')
plt.show()


print(len(pywt_out))

# Plot pywt
plt.plot(pywt_out[0])
plt.title('Pywt')
plt.show()



# V2

import numpy as np
import pywt
import matplotlib.pyplot as plt

# Generate a sample signal
t = np.linspace(0, 1, 1000, endpoint=False)
signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 25 * t)

print(signal.shape)

# Perform the CWT
num_scales = 64
scales = np.arange(1, num_scales)
coef, freqs = pywt.cwt(signal, scales, 'morl')

# Plot the results
plt.figure(figsize=(12, 8))
plt.imshow(np.abs(coef), extent=[0, 1, 1, num_scales], cmap='jet', aspect='auto', interpolation='nearest')
plt.colorbar(label='Magnitude')
plt.ylabel('Scale')
plt.xlabel('Time')
plt.title('Continuous Wavelet Transform')
plt.show()

# Optional: Plot the original signal
plt.figure(figsize=(12, 4))
plt.plot(t, signal)
plt.title('Original Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
