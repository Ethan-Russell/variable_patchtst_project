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
    def __init__(self, patch_definitions, patch_sizes, d_model):
        super().__init__()
        self.d_model = d_model
        # Store definitions to know which patches share sizes
        self.patch_definitions = patch_definitions 
        self.projs = nn.ModuleDict({
            f"p{sz}": nn.Linear(sz, d_model) for sz in patch_sizes
        })
        
        # Pre-group indices by patch size to avoid dictionary lookups in forward
        self.size_groups = {}
        for i, (_, _, sz) in enumerate(patch_definitions):
            if sz not in self.size_groups:
                self.size_groups[sz] = []
            self.size_groups[sz].append(i)

    def forward(self, list_of_patches):
        # list_of_patches: list of [B, C, P_i] tensors
        b, c, _ = list_of_patches[0].shape
        num_total_patches = len(list_of_patches)
        device = list_of_patches[0].device
        
        # 1. Pre-allocate the full output tensor to avoid torch.stack copies
        # Shape: (B*C, Num_Patches, d_model)
        embedded = torch.empty((b * c, num_total_patches, self.d_model), device=device)

        # 2. Process by Scale Group
        for sz, indices in self.size_groups.items():
            # Gather all patches of this size: (Num_in_group, B, C, sz)
            # We use torch.stack here on a smaller subset
            group_patches = torch.stack([list_of_patches[i] for i in indices], dim=0)
            
            # Flatten to (Num_in_group * B * C, sz)
            # This is one big matrix for the GPU to chew on
            group_flat = group_patches.permute(1, 2, 0, 3).reshape(-1, sz)
            
            # One single Kernel Launch per unique size
            projected = self.projs[f"p{sz}"](group_flat)
            
            # Reshape back to (B*C, Num_in_group, d_model)
            projected = projected.view(b * c, len(indices), self.d_model)
            
            # Place into the pre-allocated tensor
            embedded[:, indices, :] = projected

        return embedded
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
        
        self.num_pad = max([p[1] for p in patch_definitions]) - sequence_length

        patch_sizes = np.sort(np.unique([p[2] for p in patch_definitions]))
        num_patches = len(patch_definitions)

        print(f"Num patches: {num_patches}")
        print(f"Num pad: {self.num_pad}")

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
        self.embed = DynamicEmbedding(patch_definitions, self.patch_sizes, self.d_model)

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

        # pad with last value num_pad times
        if self.num_pad > 0:
            # Grab the last time step: (batch, 1, channels)
            last_values = past_values[:, -1:, :]
            
            # Create the padding by repeating that last value
            # We repeat the '1' in the middle dim self.num_pad times
            padding = last_values.repeat(1, self.num_pad, 1)
            
            # 3. Concatenate along the temporal dimension (dim=1)
            past_values = torch.cat([past_values, padding], dim=1)

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
