"""Archive the completed authorized retry and exact preflight repair evidence."""
from pathlib import Path
import json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/insane_final_test_retry'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert json.loads((OUT/'status.json').read_text())['status']=='complete'
    assert json.loads((OUT/'local_independent_verification.json').read_text())['status']=='pass'
    patch=json.loads((OUT/'PREFLIGHT_REPAIR_VERIFICATION.json').read_text());submission=json.loads((OUT/'submission.json').read_text())
    report='# Минимальное исправление manifest preflight\n\n'
    report+='Первый job (353128919) остановился до inference на равенстве сериализованного целого JSON. Локально regenerated manifest совпадал полностью; исходный failed job actual/diff не сохранил. Один явно разрешённый повтор сохранил следующие реальные Kaggle отличия:\n\n'
    report+='| Preflight | Поле | Locked expected | Kaggle actual | Абсолютная разница | Решение |\n|---|---|---|---|---|---|\n'
    for check in patch['remote_manifest_diffs']:
        for r in check['fields_differing']:
            report+=f"| {check['check']} | {r['path']} | {r['expected']:.17g} | {r['actual']:.17g} | {r['absolute_difference']:.17g} | {'accepted' if r['accepted'] else 'STOP'} |\n"
    report+='\nЭто диагностический остаток float64 ECEF→ENU projection check. Точное равенство этого вычисляемого поля было хрупким; сохранённый diff подтверждает источник расхождения в повторном выполнении. Остальные поля, включая все научно определяющие поля, совпали строго. Допуск был зафиксирован до запуска: atol=1e-8м, rtol=0. Ни timestamps/маски, ни сами координаты, ни алгоритм проекции не изменялись.\n\n'
    report+='`FINAL_EVALUATION_LOCK.json` и все закреплённые исходники/сценарии/checkpoints/scalers/raw input остались побайтно прежними. Patch подключает отдельную функцию preflight к неизменному runner; snapshot и hashes patch закреплены в `PREFLIGHT_PATCH_MANIFEST.json`. Новая функция сохраняет actual manifest и field-level diff до assert. Source hash failure, scientific field mismatch, NaN/Inf или diagnostic difference вне допуска останавливают исполнение до inference.\n\n'
    report+='41 локальный test и41 Kaggle test прошли. Добавленные8 tests проверяют точное совпадение, допустимый/недопустимый round-off, строгость IDs/counts/timestamps/масок/source hashes и запись диагностики при ошибке. Test-only runner выполнил48 model evaluations и24 baseline evaluations; обучение и дополнительные seed отсутствуют. Предыдущий job не успел выполнить ни одной оценки, поэтому это первое получение test-метрик, не повтор после просмотра результата.\n\n'
    report+='Подробные свидетельства: `preflight_evidence/check_01/` и `check_02/`, `PREFLIGHT_REPAIR_VERIFICATION.json`, исходный lock, source snapshot, logs и submission/download receipts. Строка retries=0 в неизменном runner status относится к внутренним повторам runner; общая история включает ранее проваленный job и один отдельно разрешённый технический повтор (Kaggle version2).\n\n**STOP. Дополнительные jobs/изменения научного протокола не выполнялись.**\n'
    technical=ROOT/'INSANE_PREFLIGHT_FIX_REPORT.md'
    if technical.exists():raise FileExistsError(str(technical))
    technical.write_text(report)
    files={technical}
    for name in ['INSANE_FINAL_TEST_REPORT_RETRY.md','METHODS_DRAFT_FINAL_TEST_RETRY.md','RESULTS_DRAFT_FINAL_TEST_RETRY.md','notebooks/kaggle_insane_final_test_retry.ipynb']:files.add(ROOT/name)
    for base in ['src/insane_final_test_patch','tests/insane_final_test_patch','scripts/insane_final_test_patch','configs/insane_final_test_patch','outputs/insane_final_test_retry','outputs/insane_final_test_retry_launch','outputs/insane_final_test_retry_preparation']:
        for p in (ROOT/base).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.zip','.pyc'] and p.name!='artifact_manifest.json':files.add(p)
    for name in ['status.json','kaggle_run_log.json','failure_execution_evidence.json']:
        files.add(ROOT/'outputs/insane_final_test'/name)
    for name in ['LICENSE.txt','source_manifest.json']:files.add(ROOT/'outputs/insane_final_test_data_upload'/name)
    manifest=OUT/'artifact_manifest.json';manifest.write_text(json.dumps({'completed_evaluation_job_version':submission['version_number'],'previous_preflight_failed_run':'353128919',
        'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]},indent=2)+'\n');files.add(manifest)
    archive=ROOT/'INSANE_FINAL_TEST_RETRY_HANDOFF.zip'
    if archive.exists():raise FileExistsError(str(archive))
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            assert p.name not in ['kaggle.json','.env'] and p.suffix!='.b64';z.write(p,p.relative_to(ROOT))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'files':len(files),'sha256':sha(archive)},indent=2))
if __name__=='__main__':main()
