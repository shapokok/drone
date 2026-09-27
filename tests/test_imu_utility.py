import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from diagnostics.imu_utility import initial_rotation,exp_rotation,mm,mv,integrate,causal_gnss_velocity

class PhysicsTests(unittest.TestCase):
 def setup_state(self,force=np.array([0.,0.,-9.80665]),yaw=0.,v=np.array([2.,-1.,.3])):
  r=initial_rotation(force,yaw)
  return {'R0':r.tolist(),'p0':[1.,2.,3.],'v0':v.tolist(),'anchor_timestamp_s':0.,'accel_bias_mean':[0.,0.,0.],'accel_bias_median':[0.,0.,0.],'mean_gyro_xyz':[0.,0.,0.],'median_gyro_xyz':[0.,0.,0.],'a_reference_world':[0.,0.,0.]}
 def test_gravity_and_frd_sign(self):
  r=initial_rotation([0,0,-9.80665],0)
  np.testing.assert_allclose(r,np.diag([1.,-1.,-1.]),atol=1e-12)
  np.testing.assert_allclose(mv(r,np.array([0.,0,-9.80665]))+[0,0,-9.80665],0,atol=1e-12)
  self.assertAlmostEqual(np.linalg.det(r),1)
 def test_heading_north_and_gyro_right_multiplication(self):
  r=initial_rotation([0,0,-9.80665],np.pi/2);np.testing.assert_allclose(mv(r,[1,0,0]),[0,1,0],atol=1e-12)
  turned=mm(initial_rotation([0,0,-9.80665],0),exp_rotation(np.array([0,0,np.pi/2])))
  np.testing.assert_allclose(mv(turned,[1,0,0]),[0,-1,0],atol=1e-12)
 def test_irregular_dt_and_constant_velocity(self):
  t=np.array([0.,.1,.31,1.41,2.]);x=np.tile([0,0,-9.80665,0,0,0],(len(t),1));initial=self.setup_state()
  d=integrate(t,x,t,t[1:],initial,'dr_zero_bias')
  np.testing.assert_allclose(d['pred'],np.array(initial['p0'])+t[1:,None]*initial['v0'],atol=1e-12)
  self.assertGreater(d['max_integration_step_s'],1.)
 def test_constant_acceleration(self):
  t=np.arange(11)*.1;x=np.tile([1,0,-9.80665,0,0,0],(11,1));initial=self.setup_state()
  d=integrate(t,x,t,t[1:],initial,'dr_zero_bias');expected=np.array(initial['p0'])+t[1:,None]*initial['v0'];expected[:,0]+=.5*t[1:]**2
  np.testing.assert_allclose(d['pred'],expected,atol=1e-12)
 def test_future_imu_does_not_change_prefix(self):
  t=np.arange(11)*.1;x=np.tile([0.,0,-9.80665,0,0,0],(11,1));y=x.copy();y[5:]+=5
  a=integrate(t,x,t,t[1:],self.setup_state(),'dr_zero_bias');b=integrate(t,y,t,t[1:],self.setup_state(),'dr_zero_bias')
  np.testing.assert_array_equal(a['pred'][:5],b['pred'][:5]);self.assertGreater(np.linalg.norm(a['pred'][-1]-b['pred'][-1]),.1)
 def test_decomposition_identity(self):
  t=np.arange(21)*.1;x=np.tile([.2,.1,-9.3,.01,.02,.03],(21,1));initial=self.setup_state();d=integrate(t,x,t,t[1:],initial,'dr_zero_bias')
  cv=np.array(initial['p0'])+t[1:,None]*initial['v0']
  np.testing.assert_allclose(d['pred']-cv,d['force_residual_displacement']+d['orientation_gravity_displacement'],atol=1e-12)
 def test_horizontal_preserves_cv_vertical(self):
  t=np.arange(21)*.1;x=np.tile([.2,.1,-9.,0,0,0],(21,1));initial=self.setup_state();d=integrate(t,x,t,t[1:],initial,'horizontal_dr_zero_bias')
  np.testing.assert_allclose(d['pred'][:,2],3+.3*t[1:],atol=1e-12)
 def test_mean_bias_removes_static_error(self):
  t=np.arange(21)*.1;x=np.tile([0,0,-9.3,.01,.02,.03],(21,1));initial=self.setup_state();initial['accel_bias_mean']=[0,0,.50665];initial['mean_gyro_xyz']=[.01,.02,.03]
  d=integrate(t,x,t,t[1:],initial,'dr_mean_bias');np.testing.assert_allclose(d['pred'],np.array(initial['p0'])+t[1:,None]*initial['v0'],atol=1e-12)
 def test_velocity_uses_only_past_fixes(self):
  t=np.arange(100)*.1;p=np.column_stack([2*t,-t,.5*t]);a=causal_gnss_velocity(t,p,t,np.arange(100));p[50:]+=1e5;b=causal_gnss_velocity(t,p,t,np.arange(100));np.testing.assert_array_equal(a[:50],b[:50]);np.testing.assert_allclose(a[-1],[2,-1,.5])
if __name__=='__main__':unittest.main()
