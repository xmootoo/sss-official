import torch
import torch.nn as nn

class Patcher(nn.Module):
    """
    Splits the input time series into patches.
    """

    def __init__(self, patch_dim : int=16, stride : int=8):
        super(Patcher, self).__init__()
        self.patch_dim = patch_dim
        self.stride = stride

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (B, M, L). B: batch_size, M: channels, L: sequence_length.
        Returns:
            patches: tensor of shape (B, M, N, P). N: number of patches, P: patch_dim.
            patches_combined: tensor of shape (B * M, N, P). N: number of patches, P: patch_dim. This is more efficient
                              to input into the Transformer encoder, as we are applying it to channels independently, thus,
                              we can combine the batch and channel dimensions and then reshape it afterwards.
        """
        B, M, L = x.shape

        # Number of patches.
        N = int((L - self.patch_dim) / self.stride) + 2

        # Pad the time series with the last value on each channel repeated S times
        last_column = x[:, :, -1:] # index
        padding = last_column.repeat(1, 1, self.stride)
        x = torch.cat((x, padding), dim=2)

        # Extract patches
        patches = x.unfold(dimension=2, size=self.patch_dim, step=self.stride) # Unfold the input tensor to extract patches.
        patches = patches.contiguous().view(B, M, N, self.patch_dim) # Reshape the tensor to (B, M, N, P).
        patches_combined = patches.view(B * M, N, self.patch_dim) # Reshape the tensor to (B * M, N, P).

        return patches
