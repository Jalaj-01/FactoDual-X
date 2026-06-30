import torch
import torch.nn as nn
from models.attention import SelfAttention, CrossModalAttention

class FactoDualXBlock(nn.Module):
    """
    FactoDual-X Transformer Block.
    Processes spatial and spectral embeddings in parallel using SelfAttention,
    fuses them using CrossModalAttention, and processes the fused spatial representation
    through a Multi-Layer Perceptron (MLP) block.
    """
    def __init__(self, D: int, N_spa: int, N_spe: int, heads: int = 4, 
                 dk: int = 16, mlp_ratio: int = 4, dropout: float = 0.1):
        """
        Args:
            D (int): Embedding dimension.
            N_spa (int): Number of spatial tokens.
            N_spe (int): Number of spectral tokens.
            heads (int): Number of attention heads.
            dk (int): Dimension of each attention head.
            mlp_ratio (int): Ratio of MLP hidden dimension to embedding dimension.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.N_spa = N_spa
        self.N_spe = N_spe
        
        self.ln_s = nn.LayerNorm(D)
        self.ln_c = nn.LayerNorm(D)
        self.ln_cm = nn.LayerNorm(D)
        self.ln_ck = nn.LayerNorm(D)
        
        self.ssa = SelfAttention(D, heads, dk, dropout)
        self.spesa = SelfAttention(D, heads, dk, dropout)
        self.cm = CrossModalAttention(D, heads, dk, dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(D, D * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D * mlp_ratio, D),
            nn.Dropout(dropout)
        )
        self.ln_out = nn.LayerNorm(D)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z (torch.Tensor): Input token sequence of shape (B, N_spa + N_spe, D)
        Returns:
            torch.Tensor: Output token sequence of shape (B, N_spa + N_spe, D)
        """
        # Split tokens into spatial and spectral parts
        Zs = Z[:, :self.N_spa, :]
        Zc = Z[:, self.N_spa:, :]
        
        # Parallel self-attention branches
        Hs = Zs + self.ssa(self.ln_s(Zs))
        Hc = Zc + self.spesa(self.ln_c(Zc))
        
        # Cross-modal fusion (query from spatial, key/value from spectral)
        Ff = self.cm(self.ln_cm(Hs), self.ln_ck(Hc))
        
        skip = Ff + Zs
        fused_spa = self.mlp(self.ln_out(skip)) + skip
        
        # Re-concatenate spatial and spectral representations for the next layer
        return torch.cat([fused_spa, Hc], dim=1)
