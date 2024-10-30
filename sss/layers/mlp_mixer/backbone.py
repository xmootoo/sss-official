import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPMixer(nn.Module):
    def __init__(self,
                 num_patches : int,
                 input_dim : int,
                 tok_mixer_dim : int,
                 cha_mixer_dim : int,
                 num_layers : int=8,
                 dropout : float=.0) -> None:
        super(MLPMixer, self).__init__()

        """
        Args:
            num_patches (int): Number of patches.
            input_dim (int): Number of input features.
            expansion_factor (int, optional): Expansion factor for the hidden layer, relative to the input size. Defaults to 2.
            num_layers (int, optional): Number of layers. Defaults to 8.
            dropout (float, optional): Dropout rate. Defaults to .0.
        """

        self.mixers = nn.Sequential(*[
            MixerLayer(input_dim, num_patches, tok_mixer_dim, cha_mixer_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.tensor): Input tensor of shape (*, num_patches, input_dim).
        """
        x = self.mixers(x)
        return x

class MixerLayer(nn.Module):
    def __init__(self, input_dim, num_patches, tok_mixer_dim, cha_mixer_dim, dropout=.0) -> None:
        super(MixerLayer, self).__init__()

        """
        Args:
            input_dim (int): Number of input features.
            num_patches (int): Number of patches.
            tok_mixer_dim (int): Token MLP hidden dimension.
            cha_mixer_dim (int): Channel MLP hidden dimension.
            dropout (float, optional): Dropout rate. Defaults to .0.
        """

        self.tok_mixer = TokenMixer(input_dim, num_patches, tok_mixer_dim, dropout)
        self.cha_mixer = ChannelMixer(input_dim, cha_mixer_dim, dropout)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.tensor): Input tensor of shape (*, num_patches, input_dim).
        """
        x = self.tok_mixer(x)
        x = self.cha_mixer(x)
        return x

class TokenMixer(nn.Module):
    def __init__(self, input_dim, num_patches, expansion_factor, dropout) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = MLP(num_patches, expansion_factor, dropout)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.tensor): Input tensor of shape (*, num_patches, input_dim).
        """

        residual = x
        x = self.norm(x)
        x = x.transpose(-2, -1) # Transpose: (*, input_dim, num_patches) -> (*, num_patches, input_dim)

        x = self.mlp(x)

        x = x.transpose(-2,-1) # Transpose: (*, num_patches, input_dim) -> (*, input_dim, num_patches)
        out = x + residual
        return out

class ChannelMixer(nn.Module):
    def __init__(self, input_dim, expansion_factor, dropout) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = MLP(input_dim, expansion_factor, dropout)

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x (torch.tensor): Input tensor of shape (*, num_patches, input_dim).
        """
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        out = x + residual
        return out

class MLP(nn.Module):
    def __init__(self, input_dim : float , hidden_dim : float,  dropout : float=.0) -> None:
        super(MLP, self).__init__()
        """
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Hidden layer dimension.
            dropout (float, optional): Dropout rate. Defaults to .0.
        """
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.tensor): Input tensor of shape (*, input_dim).
        """
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


# Test
if __name__ == "__main__":
    batch_size = 32
    num_patches = 6
    input_dim = 4
    tok_mixer_dim = 64
    cha_mixer_dim = 128
    num_layers = 3
    dropout = 0
    x = torch.randn(batch_size, num_patches, input_dim)
    model = MLPMixer(num_patches,
                     input_dim,
                     tok_mixer_dim,
                     cha_mixer_dim,
                     num_layers,
                     dropout)
    print(model(x).shape)
