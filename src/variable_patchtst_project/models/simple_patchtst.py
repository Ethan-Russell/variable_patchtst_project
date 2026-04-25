import torch
from torch import nn
import math
import numpy as np


class Patchifier(nn.Module):
    def __init__(self, patch_definitions, num_patches, patch_sizes):
        super().__init__()
        self.patch_definitions = patch_definitions
        self.num_patches = num_patches
        self.patch_sizes = patch_sizes

    def forward(self, past_values: torch.Tensor):
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
                Input for patchification

        Returns:
            `list` of length `num_patches` patches each with potentially different patch size, 
                with shape `(batch_size, num_channels, patch_length)`
        """
        return [
            past_values[:, d[0]:d[1], :].transpose(1,2)
            for d in self.patch_definitions
        ]


class DynamicEmbedding(nn.Module):
    def __init__(self, patch_sizes, d_model):
        super().__init__()
        # Create a projection for each unique patch size you plan to use
        self.projs = nn.ModuleDict({
            f"p{sz}": nn.Linear(sz, d_model) for sz in patch_sizes
        })

    def forward(self, list_of_patches):
        # Each patch in list_of_patches: (Batch, Channels, Patch_Length)
        embedded_patches = []
        
        for patch in list_of_patches:
            b, c, p = patch.shape
            
            # Flatten Batch and Channels: (B*C, P)
            patch_reshaped = patch.reshape(b * c, p)

            # Project P -> d_model
            proj = self.projs[f"p{p}"](patch_reshaped)
            embedded_patches.append(proj)

        # Stack into (B*C, Num_Patches, d_model)
        return torch.stack(embedded_patches, dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create the position matrix
        position = torch.arange(max_len).unsqueeze(1)
        
        # Compute the division term for the sine/cosine arguments
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Initialize the PE matrix: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # register_buffer ensures this is moved to the GPU with the model 
        # but is not updated by the optimizer
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape [Batch * Channels, Num_Patches, d_model]
        """
        # x.size(1) is the Num_Patches
        # self.pe[:, :x.size(1), :] slices the first N positions
        x = x + self.pe[:, :x.size(1), :]
        
        return self.dropout(x)

class SimplePatchTST(nn.Module):
    """
    A Simple PatchTST model with options for variable patch sizing.

    For each patch length, 
    """

    def __init__(
            self,
            patch_algorithm,
            sequence_length,
            forecast_length,
            d_model = 128,
            n_heads = 8,
            n_layers = 3,
            dropout = 0.1,
    ):
        super().__init__()
        patch_definitions = patch_algorithm(sequence_length)

        patch_sizes = np.sort(np.unique([p[2] for p in patch_definitions]))
        num_patches = len(patch_definitions)

        self.num_patches = num_patches
        self.patch_sizes = patch_sizes
        self.d_model = d_model
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.forecast_length = forecast_length

        # ----------------------------------------------------------------------
        # Model layers
        # ----------------------------------------------------------------------
        
        # Patchifier takes (B,L,C) and returns list of patches each of size 
        #   (B,C,P), where P may be different
        self.patchifier = Patchifier(patch_definitions, self.num_patches, self.patch_sizes)
        
        # Build the embedding that changes from heterogeneous patch dimensions 
        # to the model dimension
        self.embed = DynamicEmbedding(self.patch_sizes, self.d_model)

        # Build positional embedding
        self.position_embed = PositionalEncoding(self.d_model, self.dropout, self.sequence_length)

        # Build the transformer layer
        encoder_layer = nn.TransformerEncoderLayer(
            self.d_model,
            n_heads,
            dim_feedforward = self.d_model * 4, # Hard coded for now
            dropout = dropout,
            batch_first = True 
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers = n_layers
        )

        self.flatten = nn.Flatten(start_dim=1)
        self.head = nn.Linear(self.num_patches * self.d_model, self.forecast_length)

    def forward(self, past_values):
        # past_values: (B, L, C)
        batch_size, _, num_channels = past_values.shape

        # P = patch length (changes per patch)
        # N = num patches
        # D = model dim
        # C = num channels
        # T = forecast length
        list_of_patches = self.patchifier(past_values)  # list of (B,C,P)
        X = self.embed(list_of_patches)                 # (BxC, D, N)
        X = self.position_embed(X)                      # (BxC, D, N)
        X = self.flatten(X)                             # (BxC, DxN)
        forecast = self.head(X)                         # (BxC, T)

        # Reshape to (B, C, T)
        forecast = forecast.reshape(batch_size, num_channels, self.forecast_length)
        
        # Transpose to get to (B, T, C)
        forecast = forecast.transpose(1,2)

        return forecast
