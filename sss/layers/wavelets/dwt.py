# V1
import torch
import numpy as np
import pywt
import ptwt  # use "from src import ptwt" for a cloned the repo

# Generate dummy time series by adding noise to a sine wave
t = np.linspace(0, 1, 100)
data = np.sin(2 * np.pi * 7 * t) + np.random.normal(0, 0.5, 100)
data_torch = torch.from_numpy(data)

wavelet = pywt.Wavelet('db32')

# compare the forward fwt coefficients
print(pywt.wavedec(data, wavelet, mode='zero', level=2))
print(ptwt.wavedec(data_torch, wavelet, mode='zero', level=2))

# invert the fwt.
print(ptwt.waverec(ptwt.wavedec(data_torch, wavelet, mode='zero'),
                   wavelet))

print(pywt.wavelist(kind='discrete'))
