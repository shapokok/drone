"""Small deterministic numerical/causality tests; never train any model."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from diagnostics.next_stage import fit_frame,apply_frame,utm32_wgs84,future_gnss_exposure,permute_imu_windows
from diagnostics.clean_outages import availability_from_blocks
from pilot_v2.data import backward_indices,build_inputs,motion_labels,fit_imu_scaler,baseline_predictions
from pilot_v2.model import CausalMotion,rollout_motion
from pilot_v2.run import execute,training_specs


class GeometryTests(unittest.TestCase):
    def test_rigid_recovery_and_no_scale(self):
        x=np.random.default_rng(1).normal(size=(30,3))
        angle=.3;r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        y=np.einsum('ni,ji->nj',x,r)+[3,-2,7]
        for kind in ['se3','se2_vertical']:
            fitted=fit_frame(x,y,kind)
            np.testing.assert_allclose(apply_frame(x,fitted),y,atol=1e-12)
            self.assertEqual(fitted['scale'],1.)

    def test_similarity_is_explicit_and_translation_only(self):
        x=np.random.default_rng(2).normal(size=(30,3));y=2*x+[1,2,3]
        f=fit_frame(x,y,'similarity_diagnostic')
        self.assertAlmostEqual(f['scale'],2)
        np.testing.assert_allclose(apply_frame(x,f),y,atol=1e-12)
        f=fit_frame(x,x+[1,2,3],'translation')
        np.testing.assert_allclose(f['translation_m'],[1,2,3],atol=1e-12)

    def test_projection_reference_points(self):
        np.testing.assert_allclose(utm32_wgs84([0],[9]),[[500000,0]],atol=1e-6)
        # Independent published dataset GPS UTM first point (rounding < 1 mm).
        np.testing.assert_allclose(utm32_wgs84([47.3843571],[8.5451784]),
                                   [[465670.706847,5247978.033438]],atol=.001,rtol=0)

    def test_future_exposure_exact_windows(self):
        t=np.arange(12.);mask=availability_from_blocks(12,[(3,9)])
        exposed,fresh=future_gnss_exposure(t,mask,t,9.,4)
        np.testing.assert_array_equal(np.flatnonzero(exposed),[8])
        np.testing.assert_array_equal(exposed,fresh)

    def test_mask_recovery_on_window_boundary_has_no_exposure(self):
        t=np.arange(12.);mask=availability_from_blocks(12,[(2,8)])
        exposed,_=future_gnss_exposure(t,mask,t,8.,4)
        self.assertFalse(exposed.any())

    def test_imu_permutation_preserves_joint_rows_and_windows(self):
        x=np.arange(48).reshape(8,6)
        for mode in ['shuffle','reverse']:
            out,order=permute_imu_windows(x,4,mode,17)
            for w in range(2):
                np.testing.assert_array_equal(out[4*w:4*w+4],x[4*w+order[w]])
            np.testing.assert_array_equal(out,permute_imu_windows(x,4,mode,17)[0])


class PilotDataTests(unittest.TestCase):
    def setUp(self):
        self.t=np.arange(80,dtype=float)
        self.et=np.arange(-.2,81,2.)
        self.data={'t':self.t,'imu':np.tile(np.arange(6.),(80,1)), 'gyro_age':np.full(80,.01),
                   'event_t':self.et,'event_position':np.column_stack([2*self.et,-self.et,.5*self.et]),
                   'gt_t':np.array([-1.,81.]),'gt_position':np.array([[-2.,1.,-.5],[162.,-81.,40.5]])}
        self.mask=availability_from_blocks(80,[(30,60)])
        self.scaler={'mean':[0.]*6,'std':[1.]*6}

    def inputs(self,data=None):
        return build_inputs(self.data if data is None else data,np.arange(80),self.mask,self.scaler,[[30.,60.]])

    def test_backward_sync_does_not_choose_future_and_rejects_no_history(self):
        np.testing.assert_array_equal(backward_indices(np.array([0.,.09,.19]),np.array([.1,.2])),[1,2])
        with self.assertRaises(ValueError):backward_indices(np.array([.1]),np.array([0.]))

    def test_scaler_train_only(self):
        a=np.arange(120.).reshape(20,6);b=a.copy();b[10:]=1e9
        self.assertEqual(fit_imu_scaler(a,10),fit_imu_scaler(b,10))

    def test_inputs_independent_of_gt_and_absolute_translation(self):
        original=self.inputs();changed=dict(self.data)
        changed['gt_position']=self.data['gt_position']+1e9
        changed['event_position']=self.data['event_position']+[1000,-2000,500]
        alternative=self.inputs(changed)
        np.testing.assert_allclose(original['gnss_features'],alternative['gnss_features'],atol=1e-5)
        np.testing.assert_array_equal(original['imu_features'],alternative['imu_features'])

    def test_prefix_inputs_unchanged_by_future_sensor_values(self):
        original=self.inputs();changed=dict(self.data)
        changed['event_position']=self.data['event_position'].copy()
        changed['event_position'][self.et>65]+=1e6
        changed['imu']=self.data['imu'].copy();changed['imu'][65:]+=100
        alternative=self.inputs(changed)
        np.testing.assert_array_equal(original['gnss_features'][:65],alternative['gnss_features'][:65])
        np.testing.assert_array_equal(original['imu_features'][:65],alternative['imu_features'][:65])

    def test_no_gnss_updates_inside_outage_and_anchor_dt(self):
        x=self.inputs()
        self.assertFalse(x['clean']['gnss_update_mask'][30:60].any())
        self.assertAlmostEqual(x['integration_dt'][30],self.t[30]-x['clean']['held_source_timestamp_s'][29],places=5)

    def test_motion_labels_invariant_to_constant_gt_origin(self):
        x=self.inputs();target,mask,_=motion_labels(self.data,x)
        changed=dict(self.data);changed['gt_position']=self.data['gt_position']+[10,20,-30]
        y,m,_=motion_labels(changed,x)
        np.testing.assert_allclose(target,y,atol=1e-6);np.testing.assert_array_equal(mask,m)
        self.assertFalse(mask[self.mask].any())

    def test_zero_velocity_head_reproduces_cv_motion(self):
        x=self.inputs();held,cv=baseline_predictions(x)
        motion=rollout_motion(torch.zeros(1,80,3),torch.tensor(x['velocity_prior'])[None],
            torch.tensor(x['integration_dt'])[None],torch.tensor(self.mask)[None]).numpy()[0]
        np.testing.assert_allclose(held+motion,cv,atol=3e-5)

    def test_training_augmentation_is_deterministic(self):
        cfg={'seed':0,'train_window_samples':1024,'train_stride_samples':512,'train_stop_index':3000,
             'outage_durations_s':[10,30,60],'train_warmup_s':30,'train_recovery_s':5}
        a=training_specs({'t':np.arange(3000)*.1},cfg,2)
        b=training_specs({'t':np.arange(3000)*.1},cfg,2)
        for x,y in zip(a,b):
            self.assertEqual(x['onset_s'],y['onset_s']);np.testing.assert_array_equal(x['available'],y['available'])


class ModelTests(unittest.TestCase):
    def test_causal_prefix_and_state_carry_with_nonzero_head(self):
        torch.set_num_threads(1);torch.manual_seed(5)
        for backbone in ['gru','lstm']:
            model=CausalMotion(backbone=backbone).eval();torch.nn.init.normal_(model.head.weight)
            i=torch.randn(1,20,8);g=torch.randn(1,20,10)
            with torch.no_grad():
                full,_=model(i,g)
                mutated=i.clone();mutated[:,10:]+=10
                altered,_=model(mutated,g)
                torch.testing.assert_close(full[:,:10],altered[:,:10])
                self.assertGreater(float((full[:,10:]-altered[:,10:]).abs().max()),1e-4)
                first,state=model(i[:,:10],g[:,:10]);second,_=model(i[:,10:],g[:,10:],state)
                torch.testing.assert_close(full,torch.cat([first,second],1))

    def test_matched_parameter_count_and_gps_only_ignores_all_imu_features(self):
        full=CausalMotion();gps=CausalMotion(ablation='gps_only')
        self.assertEqual(sum(p.numel() for p in full.parameters()),sum(p.numel() for p in gps.parameters()))
        torch.nn.init.normal_(gps.head.weight)
        i=torch.randn(1,10,8);g=torch.randn(1,10,10)
        with torch.no_grad():torch.testing.assert_close(gps(i,g)[0],gps(i*100,g)[0])

    def test_training_guard_before_any_io(self):
        with self.assertRaisesRegex(RuntimeError,'Separate user command'):
            execute('/does/not/exist','/not/raw','/not/output',run_training=False)


if __name__=='__main__':unittest.main()
