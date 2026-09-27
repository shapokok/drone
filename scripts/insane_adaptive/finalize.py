"""Verify downloaded artifact arithmetic and package; never import/run a model."""
from pathlib import Path
import json,hashlib,zipfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/insane_adaptive';LAUNCH=ROOT/'outputs/insane_adaptive_launch'

def rms(e):return float(np.sqrt(np.mean(np.sum(e*e,axis=1))))

def main():
    status=json.loads((OUT/'status.json').read_text());assert status['status']=='complete' and status['completed_trainings']==6 and not status['test_used']
    receipt=json.loads((LAUNCH/'download_receipt.json').read_text());submission=json.loads((LAUNCH/'submission.json').read_text())
    for f in receipt['files']:
        p=ROOT/f['local_path'];assert p.stat().st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'],str(p)
    for rel,h in json.loads((OUT/'source_hashes.json').read_text()).items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
    assert hashlib.sha256((ROOT/'notebooks/kaggle_insane_adaptive.ipynb').read_bytes()).hexdigest()==submission['notebook_sha256']
    d=pd.read_csv(OUT/'summary.csv');assert len(d)==99 and not d.duplicated(['scenario','method','seed']).any()
    pair=pd.read_csv(OUT/'paired_differences.csv');assert len(pair)==189
    hist=pd.read_csv(OUT/'training_history.csv');initial=json.loads((OUT/'initial_weight_hashes.json').read_text())
    for seed in [0,1,2]:
        assert initial[f'adaptive_full_seed{seed}']==initial[f'adaptive_gps_only_seed{seed}']
        order=[]
        for method in ['adaptive_full','adaptive_gps_only']:
            h=hist[(hist.method==method)&(hist.seed==seed)];assert 1<=len(h)<=30
            assert int(h.loc[h.validation_macro_relative_rmse3d_m.idxmin(),'epoch'])==status['models'][f'{method}_seed{seed}']['best_epoch']
            order.append(h.episode_order_sha256.tolist())
        n=min(map(len,order));assert order[0][:n]==order[1][:n]
    errors=[];integration_errors=[];count=0
    specs=json.loads((OUT/'outages.json').read_text())['scenarios']
    for spec in specs:
        common=None
        for p in sorted((OUT/'predictions'/spec['id']).glob('*.npz')):
            with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
            assert all(np.isfinite(v).all() for v in a.values())
            for k in ['t','gt','outage','gt_native_index','source_t','availability','anchor_gt']:
                if common is not None:np.testing.assert_array_equal(a[k],common[k])
            common=a;name=p.stem
            if '_seed' in name:method,seed=name.rsplit('_seed',1);seed=int(seed);q=d[(d.scenario==spec['id'])&(d.method==method)&(d.seed==seed)]
            else:method=name;q=d[(d.scenario==spec['id'])&(d.method==method)]
            assert len(q)==1;row=q.iloc[0];select=a['outage'] if a['outage'].any() else np.ones(len(a['t']),bool)
            assert row.n_native_gt==select.sum()
            e=a['pred']-a['gt'];metrics={'absolute_rmse3d_m':rms(e[select]),'absolute_rmse_h_m':rms(e[select,:2]),'absolute_rmse_v_m':rms(e[select,2:3])}
            if a['outage'].any():
                m=a['outage'];metrics['relative_motion_rmse3d_m']=rms((a['motion']-a['target_motion'])[m]);metrics['final_unavailable_error_m']=float(np.linalg.norm(e[m][-1]))
                start=spec['onset_s'];end=start+spec['duration_s'];assert not ((a['source_t']>=start)&(a['source_t']<end)).any()
                np.testing.assert_array_equal(m,(a['t']>=start)&(a['t']<end))
            np.testing.assert_allclose(a['pred']-a['held'],a['motion'],atol=1e-7)
            assert np.all(a['source_t']<=a['token_t']+1e-8) and np.all(a['token_t']<=a['t']+1e-8)
            if 'g_token' in a:
                assert np.all((a['g_token']>=0)&(a['g_token']<=1))
                velocity=a['g_token'][:,None]*(a['velocity_prior']+a['residual_velocity_token'])
                np.testing.assert_allclose(velocity,a['velocity_token'],atol=1e-6,rtol=1e-6)
                total=np.cumsum(velocity.astype(float)*a['integration_dt'][:,None],axis=0)
                index=np.maximum.accumulate(np.where(a['all_availability'],np.arange(len(total)),0))
                motion=np.where(a['all_availability'][:,None],0.,total-total[index])
                integration_errors.append(float(np.max(np.abs(motion-a['motion_token']))));assert integration_errors[-1]<.002
                ids=a['token_index'];native=a['motion_token'][ids]+velocity[ids]*(a['t']-a['all_token_t'][ids])[:,None]*(~a['availability'])[:,None]
                np.testing.assert_allclose(native,a['motion'],atol=2e-5,rtol=1e-6)
            for key,value in metrics.items():errors.append(abs(value-float(row[key])));assert errors[-1]<1e-6
            count+=1
    assert count==99
    for r in pair.itertuples():
        a=d[(d.scenario==r.scenario)&(d.method==r.method)&(d.seed==r.seed)].iloc[0]
        q=d[(d.scenario==r.scenario)&(d.method==r.reference)]
        if r.reference.startswith('adaptive_'):q=q[q.seed==r.seed]
        assert len(q)==1;diff=a.relative_motion_rmse3d_m-q.iloc[0].relative_motion_rmse3d_m
        assert abs(diff-r.difference_m)<1e-8
    verification={'status':'pass','prediction_files':count,'summary_rows':len(d),'paired_rows':len(pair),'downloaded_hashes_verified':len(receipt['files']),
      'max_saved_metric_arithmetic_difference':max(errors),'max_gate_integral_difference_m':max(integration_errors),
      'same_point_masks_verified':True,'paired_initial_weights_and_training_order_verified':True,'checkpoint_selection_verified':True,
      'model_forward_or_training_run':False,'raw_test_files_read':False,'scope':'small arithmetic checks of existing saved outputs; no model evaluation'}
    (OUT/'independent_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    identity=f"Kaggle version {submission['version_number']}; output run ID {', '.join(receipt['output_ids'])}; six completed trainings. Local integrity verification: {count} prediction files, maximum saved-metric difference {max(errors):.3g} m. No model forward, training or test access during verification.\n"
    report=(OUT/'INSANE_ADAPTIVE_REPORT.md').read_text().replace('(gate_traces.png)','(outputs/insane_adaptive/gate_traces.png)')
    means=pd.read_csv(OUT/'duration_summary.csv')
    executive='## Итог ограниченного эксперимента\n\n'
    executive+=identity+'\n28 tests прошли до обучения. Выполнен один Kaggle job с шестью обучениями, без повторных запусков; test не загружался и не оценивался. Прежний pilot и его результаты сохранены отдельно.\n\n'
    executive+='**A — adaptive против простых baselines:** на mars_4 adaptive_full в среднем лучше Held, CV и Damped-CV при 10/30/60 с. На mars_5 сеть хуже Held и Damped-CV при всех трёх длительностях; против CV выигрывает на 30/60 с, а на 10 с средние почти совпадают. Универсальное превосходство не подтверждено. Damped-CV: τ=5 с, выбран по 540 train-эпизодам до validation-метрик. По средним нельзя заключать, что выигран каждый эпизод; все случаи приведены ниже.\n\n'
    executive+='**B — IMU против matched gps_only:** full выигрывает 24/27 сравнений эпизод×seed, по 8/9 при каждом seed. В 6/9 эпизодов выигрывает при всех трёх seeds. Средняя ошибка ниже у full на обоих validation-полётах при каждой длительности. Это воспроизводимая внутри данного протокола польза IMU, а не доказательство generalization на независимые полёты.\n\n'
    executive+='Основная метрика: среднее episode relative-motion RMSE3D внутри outage, м. ± — sample SD трёх seed-средних, не доверительный интервал. Процент — отношение разницы средних к среднему gps_only.\n\n'
    executive+='| Outage, с | Adaptive full, м | Adaptive gps_only, м | Full−gps_only, м | Разница, % | Held, м | CV, м | Damped-CV, м |\n|---|---|---|---|---|---|---|---|\n'
    for seconds in [10,30,60]:
        a=means[means.duration_s==seconds].set_index('method');f=a.loc['adaptive_full'];g=a.loc['adaptive_gps_only'];delta=f.mean_m-g.mean_m
        executive+=f"| {seconds} | {f.mean_m:.3f} ± {f.seed_sd_m:.3f} | {g.mean_m:.3f} ± {g.seed_sd_m:.3f} | {delta:+.3f} | {100*delta/g.mean_m:+.2f} | {a.loc['held_gnss','mean_m']:.3f} | {a.loc['constant_velocity_gnss','mean_m']:.3f} | {a.loc['damped_cv','mean_m']:.3f} |\n"
    executive+='\nGate показан для всех моделей/эпизодов в `gate_traces.png` и сохранён в NPZ. Среднее g по outage у full лежит примерно в 0.356–0.499, у gps_only — 0.300–0.326. Это наблюдение о коэффициенте прогноза, не детектор остановки и не причинное объяснение выигрыша. Gate обучался вместе с residual head; отдельного fixed-g контроля нет.\n\n'
    executive+='**Ограничение гипотезы:** прежний ungated pilot доступен только для seed0, поэтому из нового эксперимента нельзя выделить чистый эффект обучаемого gate относительно ungated-модели по трём seeds. Validation уже использовалась в предыдущем pilot и снова выбирает checkpoints. ESKF не готов и не оценивался. **STOP: test, новые seeds и следующая версия не запускались и автоматически не запускаются.**\n\n'
    report=report.replace('# INSANE adaptive experiment\n\n','# INSANE adaptive experiment\n\n'+executive,1)
    (ROOT/'INSANE_ADAPTIVE_REPORT.md').write_text(report+'\n## Verified run identity\n\n'+identity)
    (ROOT/'METHODS_DRAFT.md').write_text((OUT/'METHODS_DRAFT.md').read_text()+'\n## Execution identity\n\n'+identity)
    print(json.dumps(verification,indent=2));print(pd.read_csv(OUT/'duration_summary.csv').to_string(index=False))

if __name__=='__main__':main()
