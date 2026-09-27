"""Compact causal recurrent motion model. Same parameter count for full/gps_only."""
import torch
from torch import nn


class CausalMotion(nn.Module):
    def __init__(self,hidden=32,ablation='full',backbone='gru'):
        super().__init__()
        self.ablation=ablation
        self.imu_embed=nn.Sequential(nn.Linear(8,16),nn.Tanh())
        self.gps_embed=nn.Sequential(nn.Linear(10,16),nn.Tanh())
        layer=nn.GRU if backbone=='gru' else nn.LSTM
        self.recurrent=layer(32,hidden,batch_first=True)
        self.head=nn.Linear(hidden,3)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,imu,gnss,state=None):
        if self.ablation=='gps_only':imu=torch.zeros_like(imu)
        features=torch.cat([self.imu_embed(imu),self.gps_embed(gnss)],dim=-1)
        h,state=self.recurrent(features,state)
        return self.head(h),state


def rollout_motion(delta_velocity,velocity_prior,integration_dt,availability):
    """Relative displacement since last fix. No GT or teacher forcing.

    Independent outage segments; no artificial reset within an outage. At
    availability return the position estimate is set to legal held-GNSS.
    """
    velocity=velocity_prior+delta_velocity
    increments=velocity*integration_dt[...,None]
    # Cumulative motion resets on available tokens; differentiable with torch.
    total=torch.cumsum(increments,dim=1)
    index=torch.arange(total.shape[1],device=total.device)[None].expand(total.shape[0],-1)
    last=torch.cummax(torch.where(availability,index,torch.zeros_like(index)),dim=1).values
    anchor=torch.gather(total,1,last[...,None].expand(-1,-1,3))
    return torch.where(availability[...,None],torch.zeros_like(total),total-anchor)
