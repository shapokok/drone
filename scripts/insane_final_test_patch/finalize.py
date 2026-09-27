"""Verify downloaded results and write final handoff interpretation; no inference."""
from pathlib import Path
import json,hashlib,shutil,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from insane_final_test.verify import verify
from insane_adaptive.report import table
OUT=ROOT/'outputs/insane_final_test_retry';LAUNCH=ROOT/'outputs/insane_final_test_retry_launch'

def new_text(path,text):
    if path.exists() and path.read_text()!=text:raise RuntimeError('Preserve existing file: '+str(path))
    if not path.exists():path.write_text(text)

def main():
    receipt=json.loads((LAUNCH/'download_receipt.json').read_text());submission=json.loads((LAUNCH/'submission.json').read_text())
    for f in receipt['files']:
        p=ROOT/f['local_path'];assert p.stat().st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'],str(p)
    status=json.loads((OUT/'status.json').read_text());assert status['status']=='complete' and not status['training_performed'] and status['optimizer_steps']==0
    lock=json.loads((OUT/'FINAL_EVALUATION_LOCK.json').read_text())
    for rel,h in lock['evaluation_source_sha256'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h
    assert hashlib.sha256((ROOT/'notebooks/kaggle_insane_final_test_retry.ipynb').read_bytes()).hexdigest()==submission['notebook_sha256']
    assert hashlib.sha256((OUT/'FINAL_EVALUATION_LOCK.json').read_bytes()).hexdigest()==submission['final_lock_sha256']
    for name,h in lock['checkpoint_sha256'].items():assert hashlib.sha256((ROOT/'outputs/insane_adaptive/checkpoints'/name).read_bytes()).hexdigest()==h
    # Explicitly authorized saved-array verification; no model inference, training or new evaluation.
    remote_verify=json.loads((OUT/'independent_verification.json').read_text());assert remote_verify['native_gt_and_gnss_provenance_verified_against_raw']
    # Preserve strict remote verifier; local replay uses saved GT arrays because native CSV parser arithmetic is not bit-identical.
    evidence=verify(OUT,raw=None);new_text(OUT/'local_independent_verification.json',json.dumps(evidence,indent=2)+'\n')
    d=pd.read_csv(OUT/'summary.csv');pairs=pd.read_csv(OUT/'paired_differences.csv');duration=pd.read_csv(OUT/'duration_summary.csv');flight=pd.read_csv(OUT/'per_flight_summary.csv')
    for keys,file in [(['duration_s'],'duration_summary.csv'),(['flight','duration_s'],'per_flight_summary.csv')]:
        agg=pd.read_csv(OUT/file)
        for r in agg.to_dict('records'):
            q=d[(d.duration_s>0)&(d.method==r['method'])]
            for key in keys:q=q[q[key]==r[key]]
            if r['method'].startswith('adaptive_'):
                seeds=q.groupby('seed').relative_motion_rmse3d_m.mean();mean=float(seeds.mean());sd=float(seeds.std(ddof=1));assert abs(r['seed_sd_m']-sd)<1e-8
            else:mean=float(q.relative_motion_rmse3d_m.mean());assert pd.isna(r['seed_sd_m'])
            assert abs(r['mean_m']-mean)<1e-8
    identity={'kernel':submission['kernel'],'version':submission['version_number'],'output_run_ids':receipt['output_ids'],
       'status':status['status'],'model_evaluations':status['model_evaluations'],'baseline_evaluations':status['baseline_evaluations'],
       'training_performed':False,'authorized_retry_after_failed_run':'353128919','evaluation_passes':1,'downloaded_artifacts':len(receipt['files']),'downloaded_bytes':sum(f['bytes'] for f in receipt['files'])}
    new_text(OUT/'run_identity.json',json.dumps(identity,indent=2)+'\n')
    for name in ['submission.json','download_receipt.json','kaggle_run_log.json']:
        dest=OUT/name
        if not dest.exists():shutil.copyfile(LAUNCH/name,dest)
    patch=json.loads((OUT/'PREFLIGHT_PATCH_MANIFEST.json').read_text())
    for rel,h in patch['patch_source_sha256'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h
    assert patch['original_lock_sha256']==submission['final_lock_sha256']
    assert hashlib.sha256((OUT/'PREFLIGHT_PATCH_MANIFEST.json').read_bytes()).hexdigest()==submission['patch_manifest_sha256']
    diffs=[]
    for path in sorted((OUT/'preflight_evidence').glob('check_*/manifest_field_diff.json')):
        value=json.loads(path.read_text());assert value['status']=='pass' and value['strict_mismatch_count']==0
        assert all(x['matched'] for x in value['hash_checks'])
        diffs.append({'check':path.parent.name,**value})
    assert len(diffs)==2
    diagnosis={'prior_failed_run':'353128919','original_lock_unchanged':True,'only_changed_runtime_component':'preflight manifest comparison',
       'remote_manifest_diffs':[{ 'check':x['check'],'fields_differing':x['fields_differing'],'hash_checks_passed':len(x['hash_checks'])} for x in diffs]}
    new_text(OUT/'PREFLIGHT_REPAIR_VERIFICATION.json',json.dumps(diagnosis,indent=2)+'\n')
    imu=pairs[(pairs.method=='adaptive_full')&(pairs.reference=='adaptive_gps_only')]
    win=imu.groupby('seed').better.agg(['sum','count']).reset_index();win.columns=['seed','full_wins','outage_episodes']
    n=int(imu.better.sum());total=len(imu);ep=imu.groupby('scenario').better.sum();consistent=int((ep==3).sum())
    overview='## Проверенный итог финального test\n\n'
    overview+=f"Kaggle version {identity['version']}, output run ID {', '.join(identity['output_run_ids'])}. Выполнено {identity['model_evaluations']} model×scenario и {identity['baseline_evaluations']} baseline×scenario оценок; обучения и tuning не было. Это один разрешённый повтор после preflight-only failure 353128919; до него оценок было 0. 41 tests прошли до inference. Все {evidence['prediction_files_checked']} NPZ проверены, max metric discrepancy {evidence['max_metric_absolute_discrepancy_m']:.3g} м; веса до/после совпали.\n\n"
    overview+=f"**A. Full против gps_only:** full имеет меньшую primary-ошибку в {n}/{total} episode×seed сравнениях, в {consistent}/{len(ep)} эпизодах — при всех трёх seeds.\n\n"+table(win)+'\n'
    rows=[]
    for seconds in [10,30,60]:
        q=duration[duration.duration_s==seconds].set_index('method')
        if not {'adaptive_full','adaptive_gps_only'}<=set(q.index):continue
        f=q.loc['adaptive_full'];g=q.loc['adaptive_gps_only']
        rows.append({'outage_s':seconds,'full_mean_m':f.mean_m,'full_seed_SD_m':f.seed_sd_m,'gps_only_mean_m':g.mean_m,'gps_only_seed_SD_m':g.seed_sd_m,
            'full_minus_gps_m':f.mean_m-g.mean_m,'full_minus_gps_percent':100*(f.mean_m-g.mean_m)/g.mean_m})
    overview+='Независимая проверка raw GT/GNSS выполнена на Kaggle и прошла строго. Локальный пересчёт метрик использует сохранённые GT arrays: повторное чтение raw CSV на Mac даёт sub-picometre различия координат (local_raw_reparse_comparison.json); hashes исходных CSV и native timestamps совпадают точно. Данные, scientific verifier и метрики не менялись.\n\n'
    overview+='Manifest repair: actual regenerated manifests и structured diffs сохранены в preflight_evidence/. Все научные поля/hashes совпали точно; допускаются только зафиксированные float64 round-off diagnostic projection_error_m с atol1e-8м, rtol0. Исходный lock побайтно не изменён. Подробности: PREFLIGHT_REPAIR_VERIFICATION.json.\n\n'
    overview+='Основная метрика — relative-motion RMSE3D внутри outage. SD — sample standard deviation трёх seed-средних, не CI. Отрицательная разница означает преимущество full.\n\n'+table(pd.DataFrame(rows))
    overview+='\n**B. Сравнение с baselines по каждому полёту:**\n\n'
    compact=[]
    for (seq,seconds),group in flight.groupby(['flight','duration_s']):
        a=group.set_index('method').mean_m
        compact.append({'flight':seq,'outage_s':seconds,'full_m':a['adaptive_full'],'gps_only_m':a['adaptive_gps_only'],
           'held_m':a['held_gnss'],'cv_m':a['constant_velocity_gnss'],'damped_cv_m':a['damped_cv']})
    overview+=table(pd.DataFrame(compact))+'\n'
    for seq in ['mars_6','mars_7']:
        rows_seq=[r for r in compact if r['flight']==seq]
        for baseline in ['held','cv','damped_cv']:
            lower=[str(r['outage_s']) for r in rows_seq if r['full_m']<r[baseline+'_m']]
            higher=[str(r['outage_s']) for r in rows_seq if r['full_m']>=r[baseline+'_m']]
            overview+=f"{seq}: full ниже {baseline} по среднему на длительностях {', '.join(lower) or 'нет'} с; не ниже на {', '.join(higher) or 'нет'} с.\n\n"
    overview+='**Вывод по полученным test-данным:** matched преимущество full над gps_only наблюдается на обоих полётах в средних всех трёх длительностей, но full проигрывает каждому простому baseline по среднему в каждой комбинации полёт×длительность. Поэтому перенос относительной пользы IMU внутри этой пары моделей не подтверждает практического превосходства neural navigation над простыми методами.\n\n'
    overview+='**C. Границы вывода:** знаки и величины matched contrast на этих двух test-полётах определяют, перенёсся ли validation-выигрыш. Победа над gps_only не равна победе над простыми baselines; каждый полёт показан отдельно. Три seeds не являются тремя независимыми полётами. Нет matched ungated control по трём seeds, поэтому изолированный эффект gate не установлен. Gate не интерпретируется как детектор остановки. ESKF не оценивался. Эти данные не обосновывают generalization за пределы двух test-записей, доказанную новизну или готовность к Q2.\n\n**STOP: никаких изменений модели или повторного test после просмотра результата.**\n\n'
    report=(OUT/'INSANE_FINAL_TEST_REPORT.md').read_text().replace('(gate_traces.png)','(outputs/insane_final_test_retry/gate_traces.png)')
    report=report.replace('# INSANE FINAL TEST REPORT\n\n','# INSANE FINAL TEST REPORT\n\n'+overview,1)
    new_text(ROOT/'INSANE_FINAL_TEST_REPORT_RETRY.md',report)
    methods=(OUT/'METHODS_DRAFT_FINAL_TEST.md').read_text()+'\n## Technical preflight repair and execution identity\n\nThe first job stopped before inference on brittle whole-JSON equality. One explicitly authorized retry changed only diagnostic manifest comparison: projection_error_m uses absolute tolerance1e-8m and rtol0; all scientific fields/source hashes stay exact. The original evaluation lock and scientific source files remain byte-identical; separate PREFLIGHT_PATCH_MANIFEST pins the preflight wrapper. Both regenerated manifests/diffs are saved. The first job produced zero evaluations; the retry produced the first held-out scores.\n\n`'+json.dumps(identity)+'`.\n'
    new_text(ROOT/'METHODS_DRAFT_FINAL_TEST_RETRY.md',methods)
    results=(OUT/'RESULTS_DRAFT.md').read_text();results=results.replace('# RESULTS_DRAFT — held-out INSANE test\n\n','# RESULTS_DRAFT — held-out INSANE test\n\n'+overview,1)
    new_text(ROOT/'RESULTS_DRAFT_FINAL_TEST_RETRY.md',results)
    new_text(OUT/'FINAL_INTERPRETATION.md',overview)
    print(json.dumps(identity,indent=2));print(json.dumps(evidence,indent=2));print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(compact).to_string(index=False));print(win.to_string(index=False))

if __name__=='__main__':main()
