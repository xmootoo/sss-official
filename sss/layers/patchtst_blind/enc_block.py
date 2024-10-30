import torch
import torch.nn as nn
import torch.nn.functional as F
from sss.utils.utils import *
from sss.layers.patchtst_blind import *

class SupervisedHead(nn.Module):
    def __init__(self, linear_dim, pred_len, dropout=0.0):
        super().__init__()
        """
        Flattens and applies a linear layer to each channel independently to form a prediction.
        Args:
            num_channels (int): The number of channels in the input.
            linear_dim (int): The dimension of the linear layer, should be num_patches * d_model.
            pred_len (int): The length of the forecast window.
            dropout (float): The dropout value.
        """

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(linear_dim, pred_len)

    def forward(self, x) -> torch.Tensor:
        """
        Applies a linear layer to each channel independently to form a prediction, optional dropout.
        Args:
            x (torch.Tensor): The input of shape (batch_size, num_channels, num_patches, d_model)
        Returns:
            x (torch.Tensor): The output of shape (batch_size, num_channels, pred_len).
        """
        x = self.linear(x)
        x = self.dropout(x)
        return x

class EncoderBlock(nn.Module):
    """
    Args:
        d_model: The embedding dimension.
        num_heads: The number of heads in the multi-head attention models.
        dropout: The dropout value.
        batch_first: If True, then the input and output tensors are provided as (batch, seq, feature).
        norm: The type of normalization to use. Either "batch1d", "batch2d", or "layer".
    """

    def __init__(self, d_model, d_ff, num_heads, num_channels, num_patches, attn_dropout=0.0, ff_dropout=0.0, batch_first=True,
                 norm_mode="batch1d"):
        super(EncoderBlock, self).__init__()

        # Layers
        self.attn = _MultiheadAttention(num_heads=num_heads, d_model=d_model, dropout=attn_dropout,
            batch_first=batch_first)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Dropout(ff_dropout),
                                nn.Linear(d_ff, d_model))

        # Normalization
        self.norm = Norm(norm_mode, num_channels, num_patches, d_model)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, num_patches, d_model).
        Returns:
            fc_out: Output of the transformer block, a tensor of shape (batch_size, num_patches, d_model).
        """

        # Multihead Attention -> Add & Norm
        attn_out, _ = self.attn(x, x, x)
        attn_norm = self.norm(attn_out + x) # Treat the input as the query, key and value for MHA.

        # Feedforward layer -> Add & Norm
        fc_out = self.ff(attn_norm)
        fc_norm = self.norm(fc_out + attn_out)

        return fc_norm


class _MultiheadAttention(nn.Module):
    """
    Multihead Attention mechanism from the Vanilla Transformer, with some preset parameters for the PatchTST model.
    """

    def __init__(self, num_heads:int, d_model:int, dropout=0.0, batch_first=True):
        super(_MultiheadAttention, self).__init__()

        # Layers
        self.attn = nn.MultiheadAttention(embed_dim=d_model,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          batch_first=batch_first)

    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor) -> torch.Tensor:
        """
        Args:
            Q: Query embedding of shape: (batch_size, num_patches, d_model).
            K: Key embedding of shape (batch_size, num_patches, d_model).
            V: Value embedding of shape (batch_size, num_patches, d_model).
            batch_size: The batch size.
            num_patches: The sequence length.
            d_model: The embedding dimension.
        Returns:
            x: The output of the attention layer of shape (batch_size, num_patches, d_model).
        """
        return self.attn(query=Q, key=K, value=V, need_weights=False)


if __name__ == "__main__":
    # Test: PatchTST
    batch_size, num_channels, d_model = 32, 12, 128
    d_ff = 4 * d_model
    num_heads = 4
    num_patches, pred_len = 64, 96
    num_patches = num_patches
    num_enc_layers = 3

    # Initialize the PatchTST model
    patchtst = PatchTSTBackbone(num_enc_layers, d_model, d_ff, num_heads, num_channels, num_patches, pred_len, return_head=True)

    # Initialize input
    input = torch.randn(batch_size, num_channels, num_patches, d_model) # (B, M, N, D)

    print(patchtst(input).shape) #


    # # Test: SupervisedHead
    # batch_size, num_channels, num_patches, d_model = 32, 12, 100, 512
    # pred_len = 30
    # x = torch.randn(batch_size, num_channels, num_patches, d_model)

    # # Initialize the SupervisedHead
    # supervised_head = SupervisedHead(linear_dim=num_patches*d_model, pred_len=pred_len)
    # print(supervised_head(x).shape)




    # # Test: Encoder Block
    # # Parameters
    # d_model, d_ff, num_heads, num_channels, batch_size, num_patches = 512, 2048, 8, 12, 32, 100

    # # Initialize the EncoderBlock
    # encoder_block = EncoderBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads, num_channels=num_channels,
    #                              num_patches=num_patches, norm_mode='batch1d')

    # # Create a random tensor of shape (batch_size, num_patches, d_model)
    # x = torch.randn(batch_size*num_channels, num_patches, d_model)

    # # Forward pass through the EncoderBlock
    # output = encoder_block(x)

    # print(f"Encoder block output: {output}")

    # # Check the output shape
    # assert output.shape == x.shape, f"Output shape {output.shape} does not match expected shape {x.shape}"



    # Test: _MultiheadAttention
    # # Create a tensor of shape (B, M, N, D)
    # B, M, N, D = 2, 2, 3, 3
    # x = torch.randn(B, M, N, D)

    # print(f"Original tensor: {x}")

    # # Reshape the tensor to (B*M, N, D)
    # x_reshaped = x.view(B * M, N, D)

    # print(f"Reshaped tensor: {x_reshaped}")

    # x_re_reshaped = x_reshaped.view(B, M, N, D)
    # print(f"Re-reshaped tensor: {x_re_reshaped}")
