import torch
import torch.nn as nn

class FiLMCondition(nn.Module):
    def __init__(self, cond_dim: int, feat_dim: int, mlp_ratio = 4, dropout: float = 0.0):
        super().__init__()
        h = mlp_ratio * feat_dim
        self.mlp = nn.Sequential(
                nn.Linear(cond_dim, h),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(h, 2 * feat_dim)
                )
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B,N,D]
        # cond: [B,D] or [B,1,D]

        if cond.dim() == 3:  # [B,1,D] -> [B,D]
            cond = cond.mean(dim = 1)
            #cond = cond.squeeze(1)
        B, N, D = x.shape
        gamma, beta = self.mlp(cond).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)                  # [B,1,D]，boardcast to [B,N,D]
        beta  = beta.unsqueeze(1)

        return (1 + gamma) * x + beta
