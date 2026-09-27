"""Preserve failed single-job evidence; never repair, resubmit or evaluate."""
from pathlib import Path
import json,hashlib,shutil,zipfile,datetime,platform
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/insane_final_test';LAUNCH=ROOT/'outputs/insane_final_test_launch'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fresh(path,text):
    if path.exists():raise FileExistsError('Never overwrite existing result: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
def js(path,value):fresh(path,json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def main():
    assert not OUT.exists(),'A remote result directory already exists; preserve it'
    submission=json.loads((LAUNCH/'submission.json').read_text());remote=json.loads((LAUNCH/'remote_status.json').read_text())
    evidence=json.loads((LAUNCH/'failure_execution_evidence.json').read_text());assert remote['status']=='error' and evidence['metric_or_prediction_output_paths']==[]
    events=json.loads((LAUNCH/'kaggle_run_log.json').read_text());log=''.join(x.get('data','') for x in events)
    assert 'AssertionError: Test scenario/support/hash mismatch' in log and 'Exception encountered at "In [3]"' in log
    lockpath=ROOT/'configs/insane_final_test/FINAL_EVALUATION_LOCK.json';lock=json.loads(lockpath.read_text())
    assert sha(lockpath)==submission['final_lock_sha256']
    assert sha(ROOT/'notebooks/kaggle_insane_final_test.ipynb')==submission['notebook_sha256']
    for rel,h in lock['evaluation_source_sha256'].items():assert sha(ROOT/rel)==h,rel
    for name,h in lock['checkpoint_sha256'].items():assert sha(ROOT/'outputs/insane_adaptive/checkpoints'/name)==h,name
    OUT.mkdir();(OUT/'checkpoints').mkdir()
    for name in ['FINAL_EVALUATION_LOCK.json','test_manifest.json']:shutil.copyfile(ROOT/'configs/insane_final_test'/name,OUT/name)
    frozen=ROOT/'outputs/insane_final_test_preparation/bundle/frozen'
    for name in ['config.json','scalers.json','download_manifest.json','damping_lock.json','adaptive_source_hashes.json','adaptive_status.json']:
        shutil.copyfile(frozen/name,OUT/name)
    for name in lock['checkpoint_sha256']:shutil.copyfile(frozen/'checkpoints'/name,OUT/'checkpoints'/name)
    for rel in lock['evaluation_source_sha256']:
        p=OUT/'source_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p)
    js(OUT/'source_hashes.json',lock['evaluation_source_sha256'])
    for name in ['kaggle_run_log.json','submission.json','remote_status.json','download_receipt.json','failure_execution_evidence.json']:
        shutil.copyfile(LAUNCH/name,OUT/name)
    fresh(OUT/'kaggle_run_log.txt',log)
    status={'status':'failed_preflight','origin':'local classification of downloaded Kaggle error log; remote runner never created status.json',
      'kernel':submission['kernel'],'kaggle_version':submission['version_number'],'output_run_ids':evidence['output_run_ids'],
      'error':'AssertionError: Test scenario/support/hash mismatch','failure_cell':'In [3] / preflight',
      'failure_source':'src/insane_final_test/run.py:48','model_evaluations':0,'baseline_evaluations':0,'training_performed':False,
      'optimizer_steps':0,'seeds_evaluated':[],'planned_seeds':[0,1,2],'planned_flights':['mars_6','mars_7'],
      'test_metrics_available':False,'job_attempts':1,'retry_started':False,'checks_bypassed':False,
      'local_unit_tests_passed':33,'kaggle_unit_tests_executed':False,'scientific_parameters_changed_after_submission':False,
      'exact_mismatching_manifest_field':'not recorded by remote preflight; unknown','classified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    js(OUT/'status.json',status)
    js(OUT/'independent_verification.json',{'status':'not_run_no_predictions','prediction_files_checked':0,'summary_rows':0,'paired_rows':0,
       'remote_output_prediction_or_metric_files':evidence['metric_or_prediction_output_paths'],'checkpoint_file_hashes_verified':6,
       'source_and_notebook_hashes_unchanged':True,'lock_sha256_matches_submission':True,
       'weights_after_inference_verification':'not applicable; inference did not start',
       'note':'Integrity checks describe frozen inputs only; no numerical test result or metric verification exists.'})
    js(OUT/'environment.json',{'status':'not_collected_by_remote_runner','reason':'preflight failed before evaluation runner and environment export',
       'requested_platform':'Kaggle GPU','actual_gpu_model':None,'actual_torch_version':None,'local_packaging_python':platform.python_version()})
    js(OUT/'eskf_status.json',{'status':'not_ready','evaluated':False})
    runid=', '.join(evidence['output_run_ids'])
    report=f'''# INSANE FINAL TEST REPORT — остановлен на preflight

**Финальная оценка не выполнена.** Единственный разрешённый Kaggle job завершился технической ошибкой до inference. Повторный job не запускался; проверки не обходились и код/параметры после отправки не менялись.

## Статус и выполненный объём

- Kernel: `{submission['kernel']}`, version **{submission['version_number']}**, output run ID **{runid}**.
- Фактически выполнено: **0 model evaluations, 0 baseline evaluations, 0 trainings, 0 optimizer steps**.
- Планировалось: 48 model×scenario +24 baseline×scenario, всего72; шесть моделей на8 сценариях.
- Локально прошли33 tests и точная проверка notebook input с совместимостью шести checkpoints. На Kaggle выполнение остановилось в input preflight, **до remote unit tests и до test runner**.
- Полноценные оценки на Mac не выполнялись. Численных test-результатов нет.

## Точный блокер

В notebook cell `In [3]`, `src/insane_final_test/run.py:48`:

```python
assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True), 'Test scenario/support/hash mismatch'
```

Ошибка: `AssertionError: Test scenario/support/hash mismatch`.

До этой строки прошли сверки зафиксированных файлов source/config/scalers/checkpoints/raw/calibration и SHA256 файла test manifest. Затем сравнивался весь заново сформированный manifest, включая scenarios, point/mask hashes и inventory, с локально зафиксированным JSON.

**Конкретное несовпавшее поле не установлено:** реализация остановилась без экспорта actual manifest или структурированного diff. Сообщение само по себе не доказывает изменения raw data, outage-масок или GT timestamps. Разницу floating-point/inventory между средами можно рассматривать только как неподтверждённую гипотезу, не как найденную причину. Никакого ослабления равенства или исправления после сбоя не выполнялось.

Доказательства: `kaggle_run_log.json`, читаемая копия `kaggle_run_log.txt`, `failure_execution_evidence.json`, `submission.json`, source snapshot. Kaggle API перечислил105 output files подготовки, но ни одного пути `outputs/insane_final_test/`: научный runner не начал запись результатов. Файлы в этой папке созданы локально для фиксации сбоя; `status.json` явно помечает это происхождение.

## Зафиксированный протокол, ещё не оценённый

Все шесть ранее выбранных на validation best checkpoints сохранены побайтно: adaptive_full и adaptive_gps_only, seeds0/1/2. Основная модель — full; исключения seeds/ensemble нет. Scalers остались train-only, config и архитектура не менялись. Damped-CV τ=5 с. ESKF not_ready, без численных оценок.

Test: только mars_6/mars_7. Chronological/support-only selector проверен против исходного validation-алгоритма. History20s, recovery10s, onset stride60s, durations10/30/60s, прежние gaps/causal features/native GT timestamps. Зафиксированы6 outages и2controls; unsupported=[]; manifest создан до любых predictions/baseline metrics.

| Полёт | Onset, с от IMU epoch | Outage, с | Исходных GT точек внутри outage |
|---|---:|---:|---:|
| mars_6 |20.36518530845642|10|43|
| mars_6 |20.36518530845642|30|127|
| mars_6 |20.36518530845642|60|258|
| mars_7 |21.3|10|47|
| mars_7 |21.3|30|122|
| mars_7 |21.3|60|241|

Это **метаданные поддержки**, не ошибки моделей. Controls содержат392 и393 исходных GT timestamps соответственно. Длительности не менялись, полёты не склеивались, padding не считался измерениями.

## A. Подтвердилось ли преимущество full над gps_only на test?

**Проверить не удалось.** Ни один test-прогноз не выполнен. Для10/30/60с нет full/gps_only RMSE, mean/SD, paired difference или win count. Нули в количестве выполненных оценок не являются нулевыми ошибками моделей. Validation-результаты не подставляются вместо test.

## B. Сравнение с Held/CV/Damped-CV по каждому полёту

**Недоступно для обоих полётов и всех длительностей:** baseline evaluation не началась. τ=5с фиксировано, но test-ошибка Damped-CV неизвестна.

## C. Научные выводы и ограничения

Этот job не добавляет эмпирических свидетельств в пользу или против IMU/adaptive-механизма. Прежние validation-выводы остаются validation-выводами; переноса на отложенные полёты пока не показано. Результаты проверки байтов/локальных tests не заменяют научную оценку.

Не проводится изолированный трёхseed-контроль ungated, gate не является доказанным детектором остановки, ESKF не оценивался. Нельзя заявлять generalization, превосходство над INS/EKF, новизну или готовность к Q2 по этому незавершённому этапу.

## Сохранено и отсутствует

Сохранены FINAL_EVALUATION_LOCK, test manifest, config/scalers, исходные hashes/provenance, шесть неизменных checkpoints, source snapshot, notebook, логи, upload/submission receipts и классификация сбоя.

**Не получены и не сфабрикованы:** summary.csv, paired_differences.csv, duration/per-flight/per-seed test summaries, predictions/GT/masks/g(t), gate figure, фактическое remote hardware environment, test-metric verification. `independent_verification.json` сообщает not_run_no_predictions. Remote regenerated actual manifest не был экспортирован, поэтому точное несовпавшее поле неизвестно.

**STOP.** Автоматического исправления, повторного test или следующей версии нет. Исполняемые исходники и notebook оставлены в отправленном состоянии для разбора ошибки; запускать их повторно в рамках этого разрешения нельзя.
'''
    fresh(OUT/'INSANE_FINAL_TEST_REPORT.md',report);fresh(ROOT/'INSANE_FINAL_TEST_REPORT.md',report)
    methods=f'''# METHODS_DRAFT — final test stage (attempted, not executed)

This separate version preserves the previous METHODS_DRAFT.md unchanged. The preceding six adaptive trainings and their validation results remain documented there; no additional model training occurred.

## Locked final-test protocol

All six validation-selected best checkpoints of adaptive_full/adaptive_gps_only, seeds0/1/2, were frozen by SHA256 before inference. The existing causal GRU architecture, gate, input preprocessing, native GT scoring, actual-dt integration and train-only scalers were retained. Damped-CV τ remained5s. The planned test set was only mars_6/mars_7, using existing raw files and calibration hashes in a separate private Kaggle input. Neither seed selection nor ensemble nor test tuning was allowed.

The selector reused the validation chronological/support algorithm:20s permitted history,10s recovery,60s onset stride and10/30/60s outages, with unchanged gap limits. It produced one onset per test flight (six outages) plus two full-GNSS controls. No scenario was chosen by movement or prediction error. Primary was to remain native-timestamp relative-motion RMSE3D within outage, with absolute3D/H/V, final unavailable and recovery errors. All methods were to use identical points/masks, no GT attitude/RTK/onboard pose inputs or true-state resets. Episode means were to be summarized within seed then mean/sample SD across seeds, separately by flight; controls would have N/A outage metrics. Validation/test would not be pooled.

## What actually executed

One Kaggle job was submitted: `{submission['kernel']}`, version{submission['version_number']}, output run ID{runid}. Local33tests and exact-package checkpoint compatibility checks passed. On Kaggle, source/input hashes and the saved manifest hash were checked, but the regenerated test-manifest JSON failed exact equality in preflight before remote tests, checkpoint loading in that cell, model inference, baselines or scientific output export. Thus0model and0baseline evaluations completed. No test metric was produced, no weight updated, and no retry occurred.

The exact differing manifest field is not known because the failed code did not export actual/diff. A platform-dependent numeric difference is an unverified hypothesis only. The submitted source and notebook remain unchanged; checks were not weakened. The failure is documented rather than described as completed held-out evaluation.

## Implications

There is no new held-out evidence for generalization or IMU benefit from this attempt. Existing validation estimates cannot substitute for test. No claim about an isolated learned-gate effect, stop detection, INS/EKF superiority, statistical significance or Q2 readiness is justified. ESKF remains not_ready. The final-test stage stops pending a separately authorized resolution; this document does not authorize a retry.
'''
    fresh(OUT/'METHODS_DRAFT_FINAL_TEST.md',methods);fresh(ROOT/'METHODS_DRAFT_FINAL_TEST.md',methods)
    results=f'''# RESULTS_DRAFT — final test unavailable

The single final evaluation job (`{submission['kernel']}`, version{submission['version_number']}, run ID{runid}) failed during test-manifest preflight. Completed model evaluations:0; baseline evaluations:0. No full/gps_only, Held, CV or Damped-CV test metrics were calculated. All10/30/60s comparisons on both mars_6/mars_7 are unavailable, not zero.

| Test flight | Outages | full/gps_only | Held/CV/Damped-CV |
|---|---|---|---|
| mars_6 |10/30/60s|Not evaluated|Not evaluated|
| mars_7 |10/30/60s|Not evaluated|Not evaluated|

The error was `AssertionError: Test scenario/support/hash mismatch` at exact equality of locally locked and remotely regenerated manifests. The precise differing field was not recorded. This technical failure neither confirms nor refutes the scientific hypothesis. Validation findings remain separate and are not reported here as test results. There are no test seed means, SDs, paired comparisons, gate traces or trajectory metrics to interpret. No training, tuning, changed parameters or retry followed the failure. See INSANE_FINAL_TEST_REPORT.md for evidence and limitations.
'''
    fresh(OUT/'RESULTS_DRAFT.md',results);fresh(ROOT/'RESULTS_DRAFT.md',results)
    files=set()
    for name in ['INSANE_FINAL_TEST_REPORT.md','METHODS_DRAFT_FINAL_TEST.md','RESULTS_DRAFT.md','notebooks/kaggle_insane_final_test.ipynb']:files.add(ROOT/name)
    for base in ['src/insane_final_test','tests/insane_final_test','scripts/insane_final_test','configs/insane_final_test','outputs/insane_final_test','outputs/insane_final_test_launch']:
        for p in (ROOT/base).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.zip']:files.add(p)
    for name in ['preflight.json','insane_tests.log','insane_adaptive_tests.log','insane_final_test_tests.log','transport.json','upload_attempt.json','upload_receipt.json']:
        files.add(ROOT/'outputs/insane_final_test_preparation'/name)
    for name in ['source_manifest.json','LICENSE.txt','README.md','dataset-metadata.json']:files.add(ROOT/'outputs/insane_final_test_data_upload'/name)
    # Original license retained; raw sensor input and authentication are excluded.
    index=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]
    js(OUT/'artifact_manifest.json',{'stage_status':'failed_preflight','contains_test_predictions':False,'files':index});files.add(OUT/'artifact_manifest.json')
    archive=ROOT/'INSANE_FINAL_TEST_HANDOFF.zip'
    if archive.exists():raise FileExistsError('Never overwrite a handoff')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            assert p.name not in ['kaggle.json','.env'] and p.suffix!='.b64';z.write(p,p.relative_to(ROOT))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'status':status,'handoff':str(archive),'bytes':archive.stat().st_size,'files':len(files),'sha256':sha(archive)},indent=2,ensure_ascii=False))

if __name__=='__main__':main()
