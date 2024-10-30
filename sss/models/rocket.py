import torch
import torch.nn as nn

from sss.layers.rocket.random_kernels import generate_kernels, apply_kernels

class Rocket(nn.Module):
    def __init__(self, max_dilation,
                       num_kernels,
                       pred_len,
                       norm_mode="layer",
                       seed=1995):
        super(Rocket, self).__init__()

        # Parameters
        self.num_kernels = num_kernels
        self.max_dilation = max_dilation
        self.pred_len = pred_len
        self.d_model = num_kernels*2

        # Random convolutional kernels
        self.kernels = generate_kernels(max_dilation, num_kernels, seed)

        # Normalization
        if norm_mode=="layer":
            self.norm = nn.LayerNorm(self.d_model)
        elif norm_mode=="batch1d":
            self.norm = nn.BatchNorm1d(self.d_model)
        else:
            self.norm = nn.Identity()

        # Linear head
        self.head = nn.Linear(self.d_model, pred_len)

    def forward(self, x):
        B = len(x)
        device = next(self.parameters()).device
        kernel_features = torch.zeros(B, self.d_model).to(device)

        for i in range(B):
            input = x[i].unsqueeze(0).cpu().numpy()
            kernel_out = apply_kernels(x[i].unsqueeze(0).cpu().numpy(), self.kernels)
            kernel_features[i] = torch.from_numpy(kernel_out).squeeze().to(device)

        # Normalization
        out = self.norm(kernel_features)

        # Linear head
        out = self.head(out)

        return out


# Test
if __name__ == "__main__":
    # Create a list of variable-length 1D numpy arrays
    import numpy as np
    import random
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = [np.random.normal(0, 1, (random.randint(100, 1000))) for _ in range(10)]
    X = [np.expand_dims(x, 0) for x in X]

    # Print shapes of X
    print(f"Original shapes: {[x.shape for x in X]}")

    # Parameters
    max_dilation = 50
    num_kernels = 1000
    pred_len = 5
    norm_mode = "layer"

    # Create the model
    model = Rocket(max_dilation, num_kernels, pred_len, norm_mode).to(device)
    out = model(X)
    print(f"Output shape: {out.shape}")
