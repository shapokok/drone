"""Dual-branch cross-attention fusion Transformer (proposed method).

One branch encodes the IMU stream (accel+gyro, always present, full rate).
One branch encodes the GPS stream (position + an explicit availability
mask channel, so "GPS missing" is a signal the model sees, not silently
interpolated away). Cross-attention lets the IMU branch pull information
from GPS when it's available and fall back to pure inertial reasoning
when it isn't. A shared head regresses 3D ENU position at every timestep.
"""
import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CrossAttentionBlock(nn.Module):
    """query stream attends over key/value stream, residual + FFN."""

    def __init__(self, d_model, n_heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, kv, return_attn=False):
        attn_out, attn_weights = self.attn(query, kv, kv, need_weights=return_attn,
                                            average_attn_weights=True)
        x = self.norm1(query + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return (x, attn_weights) if return_attn else (x, None)


class FusionTransformer(nn.Module):
    """
    Inputs:
      imu:      (B, T, 6)  -- accel_xyz, gyro_xyz, full-rate IMU grid
      gps:      (B, T, 3)  -- ENU position resampled/held onto the IMU grid
      gps_mask: (B, T, 1)  -- 1.0 where GPS is actually available at t, else 0.0
    Output:
      pos_pred: (B, T, 3)  -- predicted ENU position at every IMU timestep
    """

    ABLATIONS = ("full", "no_cross_attn", "imu_only", "gps_only")

    def __init__(self, d_model=128, n_heads=4, n_layers=2, dropout=0.1, ablation="full"):
        super().__init__()
        assert ablation in self.ABLATIONS, ablation
        self.ablation = ablation

        self.imu_embed = nn.Linear(6, d_model)
        self.gps_embed = nn.Linear(4, d_model)  # position (3) + mask (1)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        self.imu_self_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4,
                                        dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        if ablation == "no_cross_attn":
            # same parameter budget in the fusion step, no attention: a
            # concat+linear merge instead of cross-attention. Isolates what
            # cross-attention itself buys over a static fusion.
            self.concat_fuse = nn.ModuleList([
                nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.LayerNorm(d_model))
                for _ in range(n_layers)
            ])
        else:
            self.cross_layers = nn.ModuleList([
                CrossAttentionBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)
            ])

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )

    def forward(self, imu, gps, gps_mask, return_attn=False):
        if self.ablation == "gps_only":
            imu = torch.zeros_like(imu)
        gps_in = torch.cat([gps, gps_mask], dim=-1)
        if self.ablation == "imu_only":
            gps_in = torch.zeros_like(gps_in)

        h_imu = self.pos_enc(self.imu_embed(imu))
        h_gps = self.pos_enc(self.gps_embed(gps_in))

        for layer in self.imu_self_layers:
            h_imu = layer(h_imu)

        attn_maps = []
        if self.ablation == "no_cross_attn":
            for fuse in self.concat_fuse:
                h_imu = fuse(torch.cat([h_imu, h_gps], dim=-1))
        else:
            for cross in self.cross_layers:
                h_imu, attn = cross(h_imu, h_gps, return_attn=return_attn)
                if return_attn:
                    attn_maps.append(attn)

        delta = self.head(h_imu)
        # residual on top of the last-known (held) GPS position keeps the
        # regression target small and centered, easier to fit than raw ENU.
        pos_pred = gps + delta
        return (pos_pred, attn_maps) if return_attn else pos_pred


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 4, 96
    model = FusionTransformer(d_model=64, n_heads=4, n_layers=2)
    model.eval()  # dropout off so the two forward passes below are comparable

    imu = torch.randn(B, T, 6)
    gps = torch.randn(B, T, 3)
    gps_mask = (torch.rand(B, T, 1) > 0.7).float()

    out = model(imu, gps, gps_mask)
    assert out.shape == (B, T, 3), out.shape
    assert torch.isfinite(out).all()

    out2, attn_maps = model(imu, gps, gps_mask, return_attn=True)
    # need_weights=True forces a different (non-fused) attention kernel in
    # PyTorch, so this only matches to float32 kernel-switch precision.
    assert torch.allclose(out, out2, atol=1e-5)
    assert len(attn_maps) == 2 and attn_maps[0].shape[0] == B

    n_params = sum(p.numel() for p in model.parameters())

    for ablation in ("no_cross_attn", "imu_only", "gps_only"):
        m = FusionTransformer(d_model=64, n_heads=4, n_layers=2, ablation=ablation).eval()
        o = m(imu, gps, gps_mask)
        assert o.shape == (B, T, 3) and torch.isfinite(o).all(), ablation

    print(f"fusion_transformer ok | params={n_params/1e3:.1f}K | out={tuple(out.shape)} | "
          f"ablations={FusionTransformer.ABLATIONS}")
