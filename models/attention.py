import torch
import torch.nn as nn
from einops import rearrange

class SelfAttention(nn.Module):
    """
    Standard multi-head self-attention module using einops for tensor reshaping.
    """
    def __init__(self, D: int, heads: int = 4, dk: int = 16, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = dk ** -0.5
        
        self.WQ = nn.Linear(D, heads * dk, bias=False)
        self.WK = nn.Linear(D, heads * dk, bias=False)
        self.WV = nn.Linear(D, heads * dk, bias=False)
        self.out = nn.Linear(heads * dk, D)
        self.drop = nn.Dropout(dropout)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z (torch.Tensor): Input feature tensor of shape (B, N, D)
        Returns:
            torch.Tensor: Attention output tensor of shape (B, N, D)
        """
        h = self.heads
        Q = rearrange(self.WQ(Z), 'b n (h d) -> b h n d', h=h)
        K = rearrange(self.WK(Z), 'b n (h d) -> b h n d', h=h)
        V = rearrange(self.WV(Z), 'b n (h d) -> b h n d', h=h)
        
        A = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        out_rearranged = rearrange(self.drop(A) @ V, 'b h n d -> b n (h d)')
        return self.out(out_rearranged)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention module to fuse spatial (Zs) and spectral (Zc) features.
    """
    def __init__(self, D: int, heads: int = 4, dhead: int = 16, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = dhead ** -0.5
        
        self.Wq = nn.Linear(D, heads * dhead, bias=False)
        self.PhiK = nn.Linear(D, heads * dhead, bias=False)
        self.PhiV = nn.Linear(D, heads * dhead, bias=False)
        self.out = nn.Linear(heads * dhead, D)
        self.drop = nn.Dropout(dropout)

    def forward(self, Zs: torch.Tensor, Zc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Zs (torch.Tensor): Spatial branch queries of shape (B, N_spa, D)
            Zc (torch.Tensor): Spectral branch keys & values of shape (B, N_spe, D)
        Returns:
            torch.Tensor: Fused representation of shape (B, N_spa, D)
        """
        h = self.heads
        Q = rearrange(self.Wq(Zs), 'b n (h d) -> b h n d', h=h)
        K = rearrange(self.PhiK(Zc), 'b n (h d) -> b h n d', h=h)
        V = rearrange(self.PhiV(Zc), 'b n (h d) -> b h n d', h=h)
        
        M = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        out_rearranged = rearrange(self.drop(M) @ V, 'b h n d -> b n (h d)')
        return self.out(out_rearranged)
