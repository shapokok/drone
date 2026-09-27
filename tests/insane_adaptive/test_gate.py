import sys,unittest,copy
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests/insane'))
from test_pipeline import fixture
from insane.data import build_inputs,make_labels
from insane.model import integrate_motion,native_motion
from insane.metrics import baselines
from insane_adaptive.model import AdaptiveFusionNav,integrate_gated
from insane_adaptive.baselines import damped_cv,choose_tau

class GateTests(unittest.TestCase):
    def test_one_equals_original_actual_dt(self):
        torch.manual_seed(0);dv=torch.randn(2,11,3,dtype=torch.float64);prior=torch.randn_like(dv)
        dt=torch.tensor([[.1,.3,.07,.2,.5,.2,.11,.3,.1,.2,.02]]*2)
        av=torch.tensor([[True,True,False,False,False,True,True,False,False,False,True]]*2)
        a=integrate_motion(dv,prior,dt,av);b=integrate_gated(dv,torch.ones(2,11,1),prior,dt,av)
        for u,v in zip(a,b):torch.testing.assert_close(u,v)
    def test_zero_stops_without_rewind_and_recovery(self):
        dv=torch.zeros(1,7,3);prior=torch.ones_like(dv)*2
        dt=torch.tensor([[.1,.2,.3,.4,.1,.5,.2]])
        av=torch.tensor([[True,False,False,False,False,True,False]])
        gate=torch.tensor([[[1.],[1.],[1.],[0.],[0.],[1.],[0.]]])
        m,v=integrate_gated(dv,gate,prior,dt,av)
        torch.testing.assert_close(m[0,:,0],torch.tensor([0.,.4,1.,1.,1.,0.,0.]))
        self.assertTrue(torch.equal(v[0,3:5],torch.zeros(2,3)))
    def test_all_zero_equals_held(self):
        m,v=integrate_gated(torch.ones(1,9,3),torch.zeros(1,9,1),torch.ones(1,9,3),torch.ones(1,9),torch.tensor([[True]+[False]*8]))
        self.assertEqual(float(m.abs().max()),0.);self.assertEqual(float(v.abs().max()),0.)
    def test_gate_bound_init_matching_weights_and_params(self):
        for seed in [0,1,2]:
            torch.manual_seed(seed);a=AdaptiveFusionNav('adaptive_full')
            torch.manual_seed(seed);b=AdaptiveFusionNav('adaptive_gps_only')
            self.assertEqual(sum(p.numel() for p in a.parameters()),7548)
            self.assertEqual(sum(p.numel() for p in b.parameters()),7548)
            for key,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[key],rtol=0,atol=0)
            _,g,_=a(torch.randn(1,9,6),torch.randn(1,9,6),torch.randn(1,9,10))
            self.assertTrue(torch.equal(g,torch.ones_like(g)*.5))
    def test_recurrence_state_and_future_prefix(self):
        torch.manual_seed(3);m=AdaptiveFusionNav();torch.nn.init.normal_(m.gate_head.weight);torch.nn.init.normal_(m.velocity_head.weight)
        args=[torch.randn(1,19,k) for k in [6,6,10]]
        dv,g,_=m(*args);a,ga,s=m(*(v[:,:8] for v in args));b,gb,_=m(*(v[:,8:] for v in args),state=s)
        torch.testing.assert_close(dv,torch.cat([a,b],1));torch.testing.assert_close(g,torch.cat([ga,gb],1))
        alt=[v.clone() for v in args]
        for v in alt:v[:,8:]+=1e4
        d2,g2,_=m(*alt);torch.testing.assert_close(dv[:,:8],d2[:,:8]);torch.testing.assert_close(g[:,:8],g2[:,:8])
    def test_shared_inputs_survive_gps_only(self):
        m=AdaptiveFusionNav('adaptive_gps_only');torch.nn.init.normal_(m.gate_head.weight)
        args=[torch.randn(1,12,k) for k in [6,6,10]]
        _,g,_=m(*args);_,same,_=m(args[0]+999,args[1],args[2]);torch.testing.assert_close(g,same)
        _,different,_=m(args[0],args[1],args[2]+2)
        self.assertGreater(float((g-different).abs().max().detach()),1e-5)
    def test_hidden_gnss_cannot_change_gate_or_motion(self):
        s,a,sc,c=fixture();ids=np.arange(len(a['t']));x=build_inputs(s,a,ids,[[10,20]],sc,c)
        s2=copy.deepcopy(s);inside=(s['gps_t']>=10)&(s['gps_t']<20)
        s2['gps_p'][inside]+=1e6;s2['gps_cov'][inside]+=1e6
        z=build_inputs(s2,a,ids,[[10,20]],sc,c)
        m=AdaptiveFusionNav();torch.nn.init.normal_(m.gate_head.weight)
        def pred(q):
            dv,g,_=m(*(torch.tensor(q[k])[None] for k in ['imu','gnss','common']))
            motion,_=integrate_gated(dv,g,torch.tensor(q['velocity_prior'])[None],torch.tensor(q['integration_dt'])[None],torch.tensor(q['availability'])[None])
            return motion,g
        p,g=pred(x);p2,g2=pred(z);torch.testing.assert_close(p,p2);torch.testing.assert_close(g,g2)
    def test_native_readout_gate_one_cv_and_zero_hold(self):
        s,a,sc,c=fixture();x=build_inputs(s,a,np.arange(len(a['t'])),[[10,20]],sc,c)
        rt=np.arange(.13,29.9,.125);ref={'t':rt,'p':np.column_stack([rt,rt*2,rt*0])};y=make_labels(ref,x,{},c)
        for gate in [0,1]:
            m,v=integrate_gated(torch.zeros(1,len(x['t']),3),torch.full((1,len(x['t']),1),float(gate)),torch.tensor(x['velocity_prior'])[None],torch.tensor(x['integration_dt'])[None],torch.tensor(x['availability'])[None])
            z=native_motion(m[0],v[0],x,y).numpy();expected=baselines(x,y)['constant_velocity_gnss' if gate else 'held_gnss'][1]
            np.testing.assert_allclose(z,expected,atol=1e-4)
    def test_damped_analytic_integral_and_train_guard(self):
        x={'availability':np.array([False,False]),'source_t':np.array([0.,0.]),'velocity_prior':np.ones((2,3))*2,'held':np.zeros((2,3))}
        y={'token_index':np.array([0,1]),'t':np.array([1.,10.]),'outage':np.ones(2,bool),'target_motion':np.zeros((2,3))}
        p,_=damped_cv(x,y,5);np.testing.assert_allclose(p[:,0],10*(1-np.exp(-y['t']/5)))
        with self.assertRaisesRegex(ValueError,'train episodes only'):choose_tau([{'id':'bad','flight':'mars_4'}],{},[1],['mars_1'])
        tau,rows=choose_tau([{'id':'ok','flight':'mars_1'}],{'ok':(x,y)},[1,10],['mars_1']);self.assertEqual(tau,1)
    def test_tiny_backward_gate_and_velocity(self):
        m=AdaptiveFusionNav();dv,g,_=m(torch.randn(1,8,6),torch.randn(1,8,6),torch.randn(1,8,10))
        motion,_=integrate_gated(dv,g,torch.ones_like(dv),torch.ones(1,8)*.1,torch.tensor([[True]+[False]*7]))
        (motion-2).square().mean().backward()
        self.assertGreater(float(m.gate_head.bias.grad.abs().sum()),0)
        self.assertGreater(float(m.velocity_head.bias.grad.abs().sum()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

if __name__=='__main__':unittest.main()
