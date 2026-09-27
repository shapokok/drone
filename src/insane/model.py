"""Pilot dual-branch GRU. Same learned skeleton/count for full and gps_only."""
import torch
from torch import nn

class FusionNavINSANE(nn.Module):
    def __init__(self,variant='full',hidden=32):
        super().__init__()
        if variant not in ['full','gps_only']:raise ValueError(variant)
        self.variant=variant
        self.imu_branch=nn.Sequential(nn.Linear(6,16),nn.Tanh())
        self.gnss_branch=nn.Sequential(nn.Linear(6,16),nn.Tanh())
        self.common_branch=nn.Sequential(nn.Linear(10,8),nn.Tanh())
        self.gru=nn.GRU(40,hidden,batch_first=True)
        self.velocity_head=nn.Linear(hidden,3)
        nn.init.zeros_(self.velocity_head.weight);nn.init.zeros_(self.velocity_head.bias)
    def forward(self,imu,gnss,common,state=None):
        if self.variant=='gps_only':imu=torch.zeros_like(imu)
        features=torch.cat([self.imu_branch(imu),self.gnss_branch(gnss),self.common_branch(common)],dim=-1)
        out,state=self.gru(features,state)
        return self.velocity_head(out),state

def integrate_motion(dv,prior,dt,availability):
    velocity=prior+dv
    total=torch.cumsum(velocity*dt[...,None],dim=1)
    index=torch.arange(total.shape[1],device=total.device)[None].expand(total.shape[0],-1)
    last=torch.cummax(torch.where(availability,index,torch.zeros_like(index)),dim=1).values
    anchor=torch.gather(total,1,last[...,None].expand(-1,-1,3))
    return torch.where(availability[...,None],torch.zeros_like(total),total-anchor),velocity

def native_motion(motion,velocity,x,labels):
    """Causal readout at original GT timestamps. Never interpolate future prediction."""
    idx=torch.as_tensor(labels['token_index'],device=motion.device,dtype=torch.long)
    partial=torch.as_tensor(labels['partial_dt'],device=motion.device,dtype=motion.dtype)
    absent=torch.as_tensor(~x['availability'][labels['token_index']],device=motion.device)
    return motion[idx]+velocity[idx]*partial[:,None]*absent[:,None]
