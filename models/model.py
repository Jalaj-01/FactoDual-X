import torch
import torch.nn as nn
from models.embedding import PatchEmbed3D
from models.blocks import FactoDualXBlock

class FactoDualX(nn.Module):
    """
    FactoDual-X: A Parallel Factorized Dual-Branch Transformer with Cross-Modal Attention
    for Hyperspectral Image Classification.
    """
    def __init__(self, P: int, C: int, num_classes: int, B: int = 10, D: int = 64, 
                 depth: int = 4, heads: int = 4, dk: int = 16, mlp_ratio: int = 4, 
                 dropout: float = 0.1):
        """
        Args:
            P (int): Spatial patch size.
            C (int): Number of spectral channels.
            num_classes (int): Number of classes.
            B (int): Band group size.
            D (int): Embedding dimension.
            depth (int): Number of transformer blocks.
            heads (int): Number of attention heads.
            dk (int): Dimension of each attention head.
            mlp_ratio (int): Ratio of MLP hidden dimension.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.embed = PatchEmbed3D(P, C, B, D)
        
        self.blocks = nn.ModuleList([
            FactoDualXBlock(
                D=D, 
                N_spa=self.embed.N_spa, 
                N_spe=self.embed.N_spe,
                heads=heads, 
                dk=dk, 
                mlp_ratio=mlp_ratio, 
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(D)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(D, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input hyperspectral patches of shape (B, P, P, C)
        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes)
        """
        Z = self.embed(x)
        for blk in self.blocks:
            Z = blk(Z)
            
        # Select spatial tokens only, compute mean over them, normalize, and classify
        spa_tokens = self.norm(Z)[:, :self.embed.N_spa, :]
        pooled = spa_tokens.mean(dim=1)
        return self.head(self.drop(pooled))
