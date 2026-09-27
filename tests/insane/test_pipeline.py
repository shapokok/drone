import sys,unittest,json,copy
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from insane.data import integral_at,aggregate_imu,build_inputs,fit_scaler,make_labels,enu,quality_segments
from insane.model import FusionNavINSANE,integrate_motion,native_motion
from insane.metrics import baselines,score
from insane.eskf import ESKF,G,initialize_from_allowed_history

def fixture():
    t=np.arange(0,30.001,.01);gt=np.arange(0,30.01,.2)
    sensor={'imu_t':t,'imu':np.column_stack([np.sin(t),t*0,np.ones(len(t))*G,np.zeros((len(t),3))]),
       'gps_t':gt,'gps_p':np.column_stack([gt,gt*2,gt*0]),'gps_cov':np.ones((len(gt),3))}
    cfg={'gps_gap_limit_s':1.,'imu_gap_limit_s':.1,'gt_gap_limit_s':2.,'cv_history_s':5.,'token_period_s':.05}
    agg=aggregate_imu(sensor,.05);scaler={'mean':np.zeros(6),'std':np.ones(6)}
    return sensor,agg,scaler,cfg

class CausalityTests(unittest.TestCase):
    def test_weighted_aggregation_irregular(self):
        t=np.array([0.,.1,.3,.7,1.]);v=np.column_stack([t,t*2])
        actual=np.diff(integral_at(t,v,np.array([0.,.2,.6,1.])),axis=0)
        # [0.6,1]: 0.3*0.1 + 0.7*0.3 = 0.24 (left-held samples).
        np.testing.assert_allclose(actual,[[.01,.02],[.10,.20],[.24,.48]],atol=1e-12)
    def test_future_imu_no_prefix_effect(self):
        s,a,sc,c=fixture();s2=copy.deepcopy(s);s2['imu'][s2['imu_t']>15]+=1e6
        b=aggregate_imu(s2,.05);take=a['t']<=15
        np.testing.assert_array_equal(a['imu'][take],b['imu'][take])
    def test_outage_gps_position_and_covariance_do_not_leak(self):
        s,a,sc,c=fixture();ids=np.arange(len(a['t']));blocks=[[10.,20.]]
        x=build_inputs(s,a,ids,blocks,sc,c);s2=copy.deepcopy(s);mask=(s2['gps_t']>=10)&(s2['gps_t']<20)
        s2['gps_p'][mask]+=99999;s2['gps_cov'][mask]+=888
        y=build_inputs(s2,a,ids,blocks,sc,c)
        for k in ['gnss','common','held','velocity_prior','source_t','availability']:np.testing.assert_array_equal(x[k],y[k])
    def test_no_future_gps(self):
        s,a,sc,c=fixture();x=build_inputs(s,a,np.arange(len(a['t'])),[],sc,c)
        self.assertTrue(np.all(x['source_t']<=x['t']))
        s2=copy.deepcopy(s);s2['gps_p'][s['gps_t']>12]+=10000
        y=build_inputs(s2,a,np.arange(len(a['t'])),[],sc,c)
        for k in ['gnss','common','held']:np.testing.assert_array_equal(x[k][x['t']<=12],y[k][y['t']<=12])
    def test_gt_does_not_enter_inputs(self):
        import inspect
        self.assertNotIn('reference',inspect.signature(build_inputs).parameters)
        self.assertNotIn('gt',inspect.signature(build_inputs).parameters)
    def test_train_only_scaler(self):
        s,a,sc,c=fixture();b=copy.deepcopy(a);b['imu']+=1e5
        n1=fit_scaler({'train':a,'val':b},['train']);b['imu']*=99
        self.assertEqual(n1,fit_scaler({'train':a,'val':b},['train']))
    def test_shared_features_not_ablated(self):
        torch.manual_seed(7);full=FusionNavINSANE('full');gps=FusionNavINSANE('gps_only');gps.load_state_dict(full.state_dict())
        self.assertEqual(sum(p.numel() for p in full.parameters()),sum(p.numel() for p in gps.parameters()))
        with torch.no_grad():gps.velocity_head.weight.fill_(.1)
        imu=torch.randn(1,10,6);gnss=torch.randn(1,10,6);common=torch.randn(1,10,10)
        a,_=gps(imu,gnss,common);b,_=gps(imu*999,gnss,common);torch.testing.assert_close(a,b)
        d,_=gps(imu,gnss,common+1);self.assertGreater(float((d-a).abs().max().detach()),1e-5)
    def test_recurrence_prefix_and_state_carry(self):
        torch.manual_seed(8);m=FusionNavINSANE();torch.nn.init.normal_(m.velocity_head.weight)
        args=[torch.randn(1,31,k) for k in [6,6,10]]
        whole,_=m(*args);prefix,state=m(*(v[:,:13] for v in args));rest,_=m(*(v[:,13:] for v in args),state=state)
        torch.testing.assert_close(whole,torch.cat([prefix,rest],1),atol=2e-6,rtol=1e-5)
        alt=[v.clone() for v in args]
        for v in alt:v[:,14:]+=100
        perturbed,_=m(*alt);torch.testing.assert_close(whole[:,:14],perturbed[:,:14])
    def test_zero_head_matches_cv_native_gt(self):
        s,a,sc,c=fixture();x=build_inputs(s,a,np.arange(len(a['t'])),[[10,20]],sc,c)
        rt=np.arange(.13,29.9,.125);ref={'t':rt,'p':np.column_stack([rt,rt*2,rt*0])}
        y=make_labels(ref,x,{'duration_s':10},c)
        dv=torch.zeros((1,len(x['t']),3));prior=torch.tensor(x['velocity_prior'])[None]
        motion,velocity=integrate_motion(dv,prior,torch.tensor(x['integration_dt'])[None],torch.tensor(x['availability'])[None])
        pm=native_motion(motion[0],velocity[0],x,y).numpy();expected=baselines(x,y)['constant_velocity_gnss'][1]
        np.testing.assert_allclose(pm,expected,atol=1e-4)
        self.assertEqual(len(y['t']),len(rt[(rt>=x['t'][0])&(rt<=x['t'][-1])]))
        self.assertTrue(np.all(x['t'][y['token_index']]<=y['t']))
    def test_no_padding_counted_as_reference(self):
        s,a,sc,c=fixture();x=build_inputs(s,a,np.arange(100,300),[[8,10]],sc,c)
        rt=np.arange(0,31,.7);ref={'t':rt,'p':np.column_stack([rt]*3)};y=make_labels(ref,x,{},c)
        self.assertTrue(np.all(y['t']<=x['t'][-1]));self.assertTrue(np.isin(y['t'],rt).all())
    def test_projection_anchor(self):
        ref=np.array([30.5999294,34.867308,526.594]);np.testing.assert_allclose(enu(ref[None],ref),0,atol=1e-9)
        p=enu(np.array([ref+[0,.00001,0],ref+[.00001,0,0],ref+[0,0,1]]),ref)
        self.assertGreater(p[0,0],.8);self.assertGreater(p[1,1],1);self.assertAlmostEqual(p[2,2],1,places=6)
    def test_tiny_backward_smoke(self):
        torch.manual_seed(0);m=FusionNavINSANE();args=[torch.randn(2,16,k) for k in [6,6,10]]
        pred,_=m(*args);loss=(pred-1).square().mean();loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

class ESKFTests(unittest.TestCase):
    def test_stationary_gravity(self):
        e=ESKF()
        for _ in range(100):e.predict([0,0,G],[0,0,0],.01)
        np.testing.assert_allclose(e.p,0,atol=1e-12);np.testing.assert_allclose(e.v,0,atol=1e-12)
    def test_constant_velocity_and_variable_dt(self):
        e=ESKF(v=[2,-1,.3]);ts=np.resize([.01,.025,.005,.02],100)
        for dt in ts:e.predict([0,0,G],[0,0,0],dt)
        np.testing.assert_allclose(e.p,np.array([2,-1,.3])*sum(ts),atol=1e-10)
    def test_turn_mechanization(self):
        e=ESKF(v=[2,0,0]);dt=.005;w=.5
        for k in range(400):e.predict([0,2*w,G],[0,0,w],dt)
        t=400*dt;expected=np.array([2/w*np.sin(w*t),2/w*(1-np.cos(w*t)),0])
        np.testing.assert_allclose(e.p,expected,atol=2e-5)
        np.testing.assert_allclose(e.R,Rotation.from_euler('z',w*t).as_matrix(),atol=1e-12)
    def test_covariance_gnss_return_and_bias_states(self):
        e=ESKF(p=[5,2,1],ba=[.1,0,0],bg=[0,0,.01])
        for _ in range(100):e.predict([.1,0,G],[0,0,.01],.01)
        old=np.linalg.norm(e.p);e.update_gnss([0,0,0],[.1,.1,.2])
        self.assertLess(np.linalg.norm(e.p),old);self.assertGreater(np.linalg.eigvalsh(e.P).min(),-1e-10)
        np.testing.assert_allclose(e.R.T@e.R,np.eye(3),atol=1e-12)
    def test_no_course_or_gt_initialization(self):
        with self.assertRaisesRegex(RuntimeError,'no GT/course fallback'):
            initialize_from_allowed_history(np.tile([0,0,G,0,0,0],(20,1)),np.arange(20),np.zeros((20,3)))
    def test_lever_measurement(self):
        e=ESKF(lever=[1,0,0]);e.update_gnss([1,0,0],[.1]*3);np.testing.assert_allclose(e.p,0,atol=1e-12)

if __name__=='__main__':unittest.main()
