"""Dual-branch fusion, LSTM backbone -- ablation arm answering "why a
Transformer and not a recurrent model" for the paper. Same I/O contract
as FusionTransformer so train.py/evaluate.py treat both uniformly; fusion
here is a learned gate instead of cross-attention (no attention weights
to extract, so this model is not used for the Figure 3 XAI plot).
"""
import torch
import torch.nn as nn


class FusionLSTM(nn.Module):
    """
    Inputs:
      imu:      (B, T, 6)
      gps:      (B, T, 3)
      gps_mask: (B, T, 1)
    Output:
      pos_pred: (B, T, 3)
    """

    def __init__(self, hidden=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.imu_lstm = nn.LSTM(6, hidden, n_layers, batch_first=True,
                                 dropout=dropout if n_layers > 1 else 0.0)
        self.gps_lstm = nn.LSTM(4, hidden, n_layers, batch_first=True,
                                 dropout=dropout if n_layers > 1 else 0.0)
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, imu, gps, gps_mask):
        gps_in = torch.cat([gps, gps_mask], dim=-1)
        h_imu, _ = self.imu_lstm(imu)
        h_gps, _ = self.gps_lstm(gps_in)

        g = self.gate(torch.cat([h_imu, h_gps], dim=-1))
        fused = g * h_imu + (1 - g) * h_gps

        delta = self.head(fused)
        return gps + delta


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 4, 96
    model = FusionLSTM(hidden=64, n_layers=2)
    model.eval()

    imu = torch.randn(B, T, 6)
    gps = torch.randn(B, T, 3)
    gps_mask = (torch.rand(B, T, 1) > 0.7).float()

    out = model(imu, gps, gps_mask)
    assert out.shape == (B, T, 3), out.shape
    assert torch.isfinite(out).all()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"fusion_lstm ok | params={n_params/1e3:.1f}K | out={tuple(out.shape)}")
