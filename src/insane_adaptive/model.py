"""Unchanged pilot encoder + one sigmoid scalar motion gate (33 parameters)."""
import torch
from torch import nn
from insane.model import FusionNavINSANE

class AdaptiveFusionNav(FusionNavINSANE):
    def __init__(self,variant='adaptive_full',hidden=32):
        if variant not in ['adaptive_full','adaptive_gps_only']:raise ValueError(variant)
        super().__init__(variant.removeprefix('adaptive_'),hidden)
        self.gate_head=nn.Linear(hidden,1)
        nn.init.zeros_(self.gate_head.weight);nn.init.zeros_(self.gate_head.bias)
    def forward(self,imu,gnss,common,state=None):
        if self.variant=='gps_only':imu=torch.zeros_like(imu)
        features=torch.cat([self.imu_branch(imu),self.gnss_branch(gnss),self.common_branch(common)],dim=-1)
        out,state=self.gru(features,state)
        return self.velocity_head(out),torch.sigmoid(self.gate_head(out)),state

def integrate_gated(dv,gate,prior,dt,availability):
    """Prefix integral, with legal-GNSS anchors only. Zero gate stops, never rewinds.

    This cumsum is algebraically the sequential actual-dt recurrence. At the first
    unavailable token, the unchanged adapter supplies lag from last legal fix.
    At GNSS return, the decoder resumes legal-GNSS position; GRU state persists.
    """
    velocity=gate*(prior+dv)
    total=torch.cumsum(velocity*dt[...,None],dim=1)
    index=torch.arange(total.shape[1],device=total.device)[None].expand(total.shape[0],-1)
    last=torch.cummax(torch.where(availability,index,torch.zeros_like(index)),dim=1).values
    anchor=torch.gather(total,1,last[...,None].expand(-1,-1,3))
    return torch.where(availability[...,None],torch.zeros_like(total),total-anchor),velocity
