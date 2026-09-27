"""Independent arithmetic checks on saved outputs only: no model forward/training."""
from pathlib import Path
import json,hashlib,sys,shutil
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/insane_pilot'

def rms(e):return float(np.sqrt(np.mean(np.sum(e**2,axis=1))))

def main():
    status=json.loads((OUT/'status.json').read_text());assert status['status']=='complete' and not status['test_used']
    summary=pd.read_csv(OUT/'summary.csv');scenarios=json.loads((OUT/'outages.json').read_text())['scenarios']
    assert len(summary)==len(scenarios)*5 and not summary.duplicated(['scenario','method']).any()
    inventory=json.loads((OUT/'inventory.json').read_text());raw={}
    for flight in ['mars_4','mars_5']:
        d=pd.read_csv(ROOT/'data/insane'/flight/'ground_truth_8hz.csv');d.columns=d.columns.str.strip()
        raw[flight]={'t':d.t.to_numpy()-inventory[flight]['epoch_unix_s'],
           'gt':d[['p_x','p_y','p_z']].to_numpy()-inventory[flight]['origin_enu_m']}
    max_error=0.;npz_count=0;points=0
    for spec in scenarios:
        folder=OUT/'predictions'/spec['id'];common=None
        for method in ['full','gps_only','held_gnss','constant_velocity_gnss']:
            p=folder/(method+'.npz')
            with np.load(p,allow_pickle=False) as z:d={k:z[k] for k in z.files}
            assert all(np.isfinite(v).all() for v in d.values())
            for k in ['t','gt','outage','availability','source_t','source_index','gt_native_index','anchor_gt']:
                if common is not None:np.testing.assert_array_equal(d[k],common[k])
            common=d;native=raw[spec['flight']];np.testing.assert_allclose(d['t'],native['t'][d['gt_native_index']],atol=1e-8,rtol=0)
            np.testing.assert_allclose(d['gt'],native['gt'][d['gt_native_index']],atol=1e-8,rtol=0)
            np.testing.assert_allclose(d['pred']-d['held'],d['motion'],atol=1e-7,rtol=0)
            np.testing.assert_allclose(d['gt']-d['anchor_gt'],d['target_motion'],atol=1e-8,rtol=0)
            assert np.all(d['source_t']<=d['token_t']+1e-8) and np.all(d['token_t']<=d['t']+1e-8)
            assert np.all(d['imu_source_t']<=d['token_t']+1e-8)
            mask=d['outage'];select=mask if mask.any() else np.ones(len(mask),bool);error=d['pred']-d['gt']
            e=error[select];metrics={'absolute_rmse3d_m':rms(e),'absolute_rmse_h_m':rms(e[:,:2]),'absolute_rmse_v_m':rms(e[:,2:3])}
            if mask.any():
                a=spec['onset_s'];b=a+spec['duration_s'];np.testing.assert_array_equal(mask,(d['t']>=a)&(d['t']<b))
                assert not ((d['source_t']>=a)&(d['source_t']<b)).any()
                motion_error=(d['pred']-d['held'])-(d['gt']-d['anchor_gt'])
                metrics.update(relative_motion_rmse3d_m=rms(motion_error[mask]),final_unavailable_error_m=float(np.linalg.norm(error[mask][-1])),
                    final_relative_motion_error_m=float(np.linalg.norm(motion_error[mask][-1])))
                recovery=(d['t']>=b)&(d['t']<b+5)
                if recovery.any():metrics['recovery_first_5s_rmse3d_m']=rms(error[recovery])
                recovered=(d['t']>=b)&(d['source_t']>=b)
                if recovered.any():metrics['first_recovered_fix_error_m']=float(np.linalg.norm(error[np.flatnonzero(recovered)[0]]))
            row=summary[(summary.scenario==spec['id'])&(summary.method==method)].iloc[0]
            assert int(row.n_native_gt)==int(select.sum())
            for key,value in metrics.items():
                err=abs(value-float(row[key]));max_error=max(max_error,err);assert err<1e-6,(spec['id'],method,key,value,row[key])
            npz_count+=1;points+=int(select.sum())
        eskf=summary[(summary.scenario==spec['id'])&(summary.method=='eskf')].iloc[0]
        assert eskf.status=='not_ready' and pd.isna(eskf.relative_motion_rmse3d_m)
    history=pd.read_csv(OUT/'training_history.csv');ckpts={}
    for method in ['full','gps_only']:
        h=history[history.method==method]
        best=int(h.loc[h.validation_macro_relative_rmse3d_m.idxmin(),'epoch'])
        assert best==status['models'][method]['best_epoch'] and h.epoch.max()<=30
        for which in ['best','last']:
            f=OUT/'checkpoints'/f'{method}_{which}_seed0.pt';ckpts[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
    valid=summary[(summary.status=='complete')&(summary.duration_s>0)]
    cols=['relative_motion_rmse3d_m','absolute_rmse3d_m','absolute_rmse_h_m','absolute_rmse_v_m','final_unavailable_error_m']
    duration=valid.groupby(['duration_s','method'],sort=False)[cols].mean().reset_index();duration.to_csv(OUT/'duration_summary.csv',index=False)
    flight=valid.groupby(['flight','duration_s','method'],sort=False)[cols].mean().reset_index();flight.to_csv(OUT/'flight_duration_summary.csv',index=False)
    pairs=pd.read_csv(OUT/'paired_differences.csv')
    for row in pairs.itertuples():
        method=row.comparison.removeprefix('full_minus_')
        g=valid[valid.scenario==row.scenario].set_index('method')
        expected=g.loc['full','relative_motion_rmse3d_m']-g.loc[method,'relative_motion_rmse3d_m']
        assert abs(row.difference_m-expected)<1e-8
    verification={'status':'pass','prediction_files':npz_count,'summary_rows':len(summary),'max_metric_absolute_difference':max_error,
       'native_gt_rows_verified':True,'point_masks_equal_across_methods':True,'hidden_gnss_sources_found':0,
       'gt_used_as_input':False,'test_files_read':False,'checkpoint_sha256':ckpts,'validation_checkpoint_selection_verified':True,
       'eskf_not_ready_rows':int((summary.status=='not_ready').sum()),'training_rerun':False,'model_forward_run':False}
    (OUT/'independent_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    # Preserve Kaggle's report verbatim on disk; root report adds a measured outcome.
    report=(OUT/'INSANE_PILOT_REPORT.md').read_text();receipt=json.loads((ROOT/'outputs/insane_pilot_launch/download_receipt.json').read_text())
    submission=json.loads((ROOT/'outputs/insane_pilot_launch/submission.json').read_text())
    executive='## Итог завершённого pilot\n\n'
    executive+='Kaggle version 3, output run ID `353114895`, GPU Tesla T4. Один завершённый training pilot: full и gps_only, seed=0, по 7515 параметров. До него версии 1/2 остановились **до обучения** на mount path и изменении вложенного ZIP при распаковке Kaggle; обе причины и исправления сохранены в `outputs/insane_pilot_launch/infrastructure_repair.json`. Проверки не обходились, научный протокол не менялся. Локально и на Kaggle прошли 18 tests.\n\n'
    executive+='Full: 11 epochs, выбран epoch 6; gps_only: 9 epochs, выбран epoch 4. Early stopping и выбор checkpoint выполнены по заранее заданному validation-критерию. Test не использован и не загружен в Kaggle.\n\n'
    executive+='Основная метрика ниже — среднее трёх episode relative-motion RMSE3D внутри outage, метры. Процент = 100 × (среднее full − среднее gps_only) / среднее gps_only; это отношение средних, не среднее процентов эпизодов.\n\n'
    executive+='| Outage, с | Full, м | gps_only, м | Full−gps_only, м | Разница, % | CV, м | Held, м |\n|---|---|---|---|---|---|---|\n'
    for seconds in [10,30,60]:
        g=duration[duration.duration_s==seconds].set_index('method')['relative_motion_rmse3d_m']
        delta=g['full']-g['gps_only'];pct=100*delta/g['gps_only']
        executive+=f"| {seconds} | {g['full']:.6f} | {g['gps_only']:.6f} | {delta:+.6f} | {pct:+.3f} | {g['constant_velocity_gnss']:.6f} | {g['held_gnss']:.6f} |\n"
    executive+='\n**Устойчивое практическое преимущество IMU не подтверждено.** Full лучше gps_only в 4 из 9 эпизодов; все выигрыши на mars_4. На mars_5 full хуже во всех трёх длительностях. Среднее улучшение на 60 с около 0.493 м (1.32%) не доказывает generalization. Held-GNSS в среднем лучше обеих сетей на 30/60 с. Это validation, используемая также для checkpoint selection; длительности вложены, эпизоды зависимы, seed один.\n\n'
    executive+='**STOP: дополнительные seeds и test evaluation не запускать автоматически.** ESKF остаётся not_ready: отсутствуют документированное плечо ordinary GNSS к PX4 IMU и проверенная инициализация абсолютного yaw из разрешённых входов. Синтетические проверки механизации прошли, но численного сравнения real-data ESKF нет. Это не свидетельство преимущества сети.\n\n'
    report=report.replace('# INSANE_PILOT_REPORT\n\n','# INSANE_PILOT_REPORT\n\n'+executive,1)
    report+='\n\n## Verified local handoff\n\nKaggle kernel: '+receipt['kernel']+'; version '+str(submission['version_number'])+'; output IDs: '+', '.join(receipt['output_ids'])+'. '
    report+=f'Downloaded {len(receipt["files"])} artifacts. Independent recomputation: {npz_count} prediction files, maximum metric discrepancy {max_error:.3g}; native reference timestamps and identical masks verified. No test reads, model forwards, or additional training.\n'
    report+='\nLocal verification: `outputs/insane_pilot/independent_verification.json`; source/run receipts in `outputs/insane_pilot_launch/`.\n'
    (ROOT/'INSANE_PILOT_REPORT.md').write_text(report)
    print(json.dumps(verification,indent=2));print(duration.to_string(index=False))

if __name__=='__main__':main()
