import torch
import torch.nn as nn

class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding layer. 
    Processes hyperspectral patches into parallel spatial and spectral embeddings,
    adding learned positional encodings to each branch.
    """
    def __init__(self, P: int, C: int, B: int = 10, D: int = 64):
        """
        Args:
            P (int): Spatial patch size (height and width).
            C (int): Number of spectral channels.
            B (int): Band group size.
            D (int): Embedding dimension.
        """
        super().__init__()
        self.N_spa = P * P
        self.N_spe = C // B
        
        self.E_spa = nn.Linear(C, D)
        self.E_spe = nn.Linear(B * P * P, D)
        
        self.Epos_spa = nn.Parameter(torch.zeros(1, self.N_spa, D))
        self.Epos_spe = nn.Parameter(torch.zeros(1, self.N_spe, D))
        
        nn.init.trunc_normal_(self.Epos_spa, std=0.02)
        nn.init.trunc_normal_(self.Epos_spe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, P, P, C)
        Returns:
            torch.Tensor: Concatenated spatial and spectral embedding of shape (B, N_spa + N_spe, D)
        """
        B_sz, P_sz, _, C_sz = x.shape
        
        # Spatial branch embedding
        spa = x.reshape(B_sz, P_sz * P_sz, C_sz)
        Zs = self.E_spa(spa) + self.Epos_spa
        
        # Spectral branch embedding
        spe = x.permute(0, 3, 1, 2).reshape(B_sz, self.N_spe, -1)
        Zc = self.E_spe(spe) + self.Epos_spe
        
        return torch.cat([Zs, Zc], dim=1)
