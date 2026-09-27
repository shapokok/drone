# RESULTS_DRAFT — held-out INSANE test

## Проверенный итог финального test

Kaggle version 2, output run ID 353132214. Выполнено 48 model×scenario и 24 baseline×scenario оценок; обучения и tuning не было. Это один разрешённый повтор после preflight-only failure 353128919; до него оценок было 0. 41 tests прошли до inference. Все 72 NPZ проверены, max metric discrepancy 7.11e-15 м; веса до/после совпали.

**A. Full против gps_only:** full имеет меньшую primary-ошибку в 17/18 episode×seed сравнениях, в 5/6 эпизодах — при всех трёх seeds.

| seed | full_wins | outage_episodes |
|---|---|---|
| 0 | 5 | 6 |
| 1 | 6 | 6 |
| 2 | 6 | 6 |

Независимая проверка raw GT/GNSS выполнена на Kaggle и прошла строго. Локальный пересчёт метрик использует сохранённые GT arrays: повторное чтение raw CSV на Mac даёт sub-picometre различия координат (local_raw_reparse_comparison.json); hashes исходных CSV и native timestamps совпадают точно. Данные, scientific verifier и метрики не менялись.

Manifest repair: actual regenerated manifests и structured diffs сохранены в preflight_evidence/. Все научные поля/hashes совпали точно; допускаются только зафиксированные float64 round-off diagnostic projection_error_m с atol1e-8м, rtol0. Исходный lock побайтно не изменён. Подробности: PREFLIGHT_REPAIR_VERIFICATION.json.

Основная метрика — relative-motion RMSE3D внутри outage. SD — sample standard deviation трёх seed-средних, не CI. Отрицательная разница означает преимущество full.

| outage_s | full_mean_m | full_seed_SD_m | gps_only_mean_m | gps_only_seed_SD_m | full_minus_gps_m | full_minus_gps_percent |
|---|---|---|---|---|---|---|
| 10 | 1.356255 | 0.096968 | 2.054597 | 0.048054 | -0.698342 | -33.989249 |
| 30 | 8.972275 | 0.152832 | 9.907792 | 0.115160 | -0.935518 | -9.442243 |
| 60 | 17.221968 | 0.925229 | 19.324801 | 0.197217 | -2.102834 | -10.881527 |

**B. Сравнение с baselines по каждому полёту:**

| flight | outage_s | full_m | gps_only_m | held_m | cv_m | damped_cv_m |
|---|---|---|---|---|---|---|
| mars_6 | 10 | 0.812568 | 2.016123 | 0.225535 | 0.643520 | 0.365843 |
| mars_6 | 30 | 12.979377 | 13.618722 | 10.797290 | 9.284420 | 10.454532 |
| mars_6 | 60 | 19.500286 | 21.684541 | 16.673797 | 14.269841 | 16.296715 |
| mars_7 | 10 | 1.899942 | 2.093071 | 0.059445 | 0.768964 | 0.417836 |
| mars_7 | 30 | 4.965172 | 6.196863 | 1.726812 | 3.209337 | 1.982717 |
| mars_7 | 60 | 14.943649 | 16.965062 | 11.755433 | 14.515574 | 12.025496 |

mars_6: full ниже held по среднему на длительностях нет с; не ниже на 10, 30, 60 с.

mars_6: full ниже cv по среднему на длительностях нет с; не ниже на 10, 30, 60 с.

mars_6: full ниже damped_cv по среднему на длительностях нет с; не ниже на 10, 30, 60 с.

mars_7: full ниже held по среднему на длительностях нет с; не ниже на 10, 30, 60 с.

mars_7: full ниже cv по среднему на длительностях нет с; не ниже на 10, 30, 60 с.

mars_7: full ниже damped_cv по среднему на длительностях нет с; не ниже на 10, 30, 60 с.

**Вывод по полученным test-данным:** matched преимущество full над gps_only наблюдается на обоих полётах в средних всех трёх длительностей, но full проигрывает каждому простому baseline по среднему в каждой комбинации полёт×длительность. Поэтому перенос относительной пользы IMU внутри этой пары моделей не подтверждает практического превосходства neural navigation над простыми методами.

**C. Границы вывода:** знаки и величины matched contrast на этих двух test-полётах определяют, перенёсся ли validation-выигрыш. Победа над gps_only не равна победе над простыми baselines; каждый полёт показан отдельно. Три seeds не являются тремя независимыми полётами. Нет matched ungated control по трём seeds, поэтому изолированный эффект gate не установлен. Gate не интерпретируется как детектор остановки. ESKF не оценивался. Эти данные не обосновывают generalization за пределы двух test-записей, доказанную новизну или готовность к Q2.

**STOP: никаких изменений модели или повторного test после просмотра результата.**

This draft reports the locked test run only; validation values are not pooled. The primary metric is native-timestamp relative-motion RMSE3D inside outage, metres.

## All observations

| flight | scenario | method | seed | duration_s | relative_motion_rmse3d_m | absolute_rmse3d_m | absolute_rmse_h_m | absolute_rmse_v_m | final_unavailable_error_m |
|---|---|---|---|---|---|---|---|---|---|
| mars_6 | mars_6_e1_d10 | held_gnss | — | 10 | 0.225535 | 5.030611 | 4.429654 | 2.384370 | 5.595697 |
| mars_6 | mars_6_e1_d10 | constant_velocity_gnss | — | 10 | 0.643520 | 4.587604 | 4.205312 | 1.833430 | 4.575199 |
| mars_6 | mars_6_e1_d10 | damped_cv | — | 10 | 0.365843 | 4.765937 | 4.298659 | 2.058078 | 5.138968 |
| mars_6 | mars_6_e1_d10 | adaptive_full | 0.000000 | 10 | 0.639842 | 5.316423 | 4.698522 | 2.487618 | 6.556761 |
| mars_6 | mars_6_e1_d10 | adaptive_full | 1.000000 | 10 | 0.494416 | 5.256856 | 4.711026 | 2.332544 | 6.466985 |
| mars_6 | mars_6_e1_d10 | adaptive_full | 2.000000 | 10 | 1.303446 | 5.714213 | 5.421791 | 1.804554 | 6.930753 |
| mars_6 | mars_6_e1_d10 | adaptive_gps_only | 0.000000 | 10 | 2.064416 | 6.446230 | 6.118648 | 2.028800 | 8.315769 |
| mars_6 | mars_6_e1_d10 | adaptive_gps_only | 1.000000 | 10 | 2.017422 | 6.416073 | 6.076666 | 2.059155 | 8.259225 |
| mars_6 | mars_6_e1_d10 | adaptive_gps_only | 2.000000 | 10 | 1.966530 | 6.371779 | 6.026497 | 2.069035 | 8.173285 |
| mars_6 | mars_6_e1_d30 | held_gnss | — | 30 | 10.797290 | 14.227856 | 12.390379 | 6.993597 | 26.997701 |
| mars_6 | mars_6_e1_d30 | constant_velocity_gnss | — | 30 | 9.284420 | 12.563074 | 11.527801 | 4.994061 | 24.266017 |
| mars_6 | mars_6_e1_d30 | damped_cv | — | 30 | 10.454532 | 13.825744 | 12.193876 | 6.516179 | 26.518524 |
| mars_6 | mars_6_e1_d30 | adaptive_full | 0.000000 | 30 | 13.204409 | 16.879409 | 15.869406 | 5.751211 | 31.723704 |
| mars_6 | mars_6_e1_d30 | adaptive_full | 1.000000 | 30 | 12.660348 | 16.422288 | 15.219335 | 6.169552 | 30.440068 |
| mars_6 | mars_6_e1_d30 | adaptive_full | 2.000000 | 30 | 13.073373 | 16.657671 | 16.008390 | 4.605370 | 31.151045 |
| mars_6 | mars_6_e1_d30 | adaptive_gps_only | 0.000000 | 30 | 13.719452 | 17.637189 | 16.677976 | 5.737207 | 31.595606 |
| mars_6 | mars_6_e1_d30 | adaptive_gps_only | 1.000000 | 30 | 13.638908 | 17.558929 | 16.558190 | 5.843145 | 31.473050 |
| mars_6 | mars_6_e1_d30 | adaptive_gps_only | 2.000000 | 30 | 13.497806 | 17.416883 | 16.392282 | 5.885653 | 31.231575 |
| mars_6 | mars_6_e1_d60 | held_gnss | — | 60 | 16.673797 | 19.972434 | 17.797601 | 9.063306 | 15.271053 |
| mars_6 | mars_6_e1_d60 | constant_velocity_gnss | — | 60 | 14.269841 | 17.189137 | 16.290425 | 5.485299 | 11.146995 |
| mars_6 | mars_6_e1_d60 | damped_cv | — | 60 | 16.296715 | 19.545920 | 17.584847 | 8.533239 | 14.815707 |
| mars_6 | mars_6_e1_d60 | adaptive_full | 0.000000 | 60 | 19.305146 | 23.183607 | 22.020094 | 7.252247 | 20.575295 |
| mars_6 | mars_6_e1_d60 | adaptive_full | 1.000000 | 60 | 18.535354 | 22.491135 | 21.110573 | 7.758536 | 20.177546 |
| mars_6 | mars_6_e1_d60 | adaptive_full | 2.000000 | 60 | 20.660360 | 24.269370 | 23.746407 | 5.011036 | 26.414808 |
| mars_6 | mars_6_e1_d60 | adaptive_gps_only | 0.000000 | 60 | 21.888443 | 25.990045 | 25.094804 | 6.762638 | 32.653690 |
| mars_6 | mars_6_e1_d60 | adaptive_gps_only | 1.000000 | 60 | 21.731647 | 25.839739 | 24.889606 | 6.942593 | 32.310088 |
| mars_6 | mars_6_e1_d60 | adaptive_gps_only | 2.000000 | 60 | 21.433532 | 25.543171 | 24.557210 | 7.028299 | 31.683549 |
| mars_6 | mars_6_control | held_gnss | — | 0 | — | 4.053204 | 3.498657 | 2.046426 | — |
| mars_6 | mars_6_control | constant_velocity_gnss | — | 0 | — | 4.053204 | 3.498657 | 2.046426 | — |
| mars_6 | mars_6_control | damped_cv | — | 0 | — | 4.053204 | 3.498657 | 2.046426 | — |
| mars_6 | mars_6_control | adaptive_full | 0.000000 | 0 | — | 4.053204 | 3.498657 | 2.046426 | — |
| mars_6 | mars_6_control | adaptive_full | 1.000000 | 0 | — | 4.053204 | 3.498657 | 2.046426 | — |
| mars_6 | mars_6_control | adaptive_full | 2.000000 | 0 | — | 4.053204 | 3.498657 | 2.046426 | — |
| mars_6 | mars_6_control | adaptive_gps_only | 0.000000 | 0 | — | 4.053204 | 3.498657 | 2.046426 | — |
| mars_6 | mars_6_control | adaptive_gps_only | 1.000000 | 0 | — | 4.053204 | 3.498657 | 2.046426 | — |
| mars_6 | mars_6_control | adaptive_gps_only | 2.000000 | 0 | — | 4.053204 | 3.498657 | 2.046426 | — |
| mars_7 | mars_7_e1_d10 | held_gnss | — | 10 | 0.059445 | 1.521070 | 1.385516 | 0.627694 | 1.528813 |
| mars_7 | mars_7_e1_d10 | constant_velocity_gnss | — | 10 | 0.768964 | 1.511978 | 1.477898 | 0.319214 | 1.837614 |
| mars_7 | mars_7_e1_d10 | damped_cv | — | 10 | 0.417836 | 1.462931 | 1.424993 | 0.331002 | 1.551552 |
| mars_7 | mars_7_e1_d10 | adaptive_full | 0.000000 | 10 | 2.229051 | 3.350971 | 3.246363 | 0.830742 | 4.221225 |
| mars_7 | mars_7_e1_d10 | adaptive_full | 1.000000 | 10 | 2.001087 | 2.869968 | 2.823376 | 0.515042 | 3.413359 |
| mars_7 | mars_7_e1_d10 | adaptive_full | 2.000000 | 10 | 1.469686 | 2.834093 | 2.685524 | 0.905562 | 3.766429 |
| mars_7 | mars_7_e1_d10 | adaptive_gps_only | 0.000000 | 10 | 2.139733 | 3.037441 | 2.967149 | 0.649674 | 4.253047 |
| mars_7 | mars_7_e1_d10 | adaptive_gps_only | 1.000000 | 10 | 2.094038 | 2.986741 | 2.920779 | 0.624236 | 4.169735 |
| mars_7 | mars_7_e1_d10 | adaptive_gps_only | 2.000000 | 10 | 2.045442 | 2.936317 | 2.874937 | 0.597239 | 4.084101 |
| mars_7 | mars_7_e1_d30 | held_gnss | — | 30 | 1.726812 | 2.742962 | 2.579837 | 0.931818 | 4.387730 |
| mars_7 | mars_7_e1_d30 | constant_velocity_gnss | — | 30 | 3.209337 | 3.633219 | 2.999768 | 2.049798 | 7.246716 |
| mars_7 | mars_7_e1_d30 | damped_cv | — | 30 | 1.982717 | 2.827831 | 2.665273 | 0.944959 | 4.780636 |
| mars_7 | mars_7_e1_d30 | adaptive_full | 0.000000 | 30 | 5.085568 | 6.212921 | 6.093301 | 1.213289 | 8.792760 |
| mars_7 | mars_7_e1_d30 | adaptive_full | 1.000000 | 30 | 5.174257 | 5.977551 | 5.854862 | 1.204872 | 9.511223 |
| mars_7 | mars_7_e1_d30 | adaptive_full | 2.000000 | 30 | 4.635692 | 5.893001 | 5.769785 | 1.198769 | 8.919758 |
| mars_7 | mars_7_e1_d30 | adaptive_gps_only | 0.000000 | 30 | 6.312595 | 7.113220 | 7.051573 | 0.934457 | 13.357061 |
| mars_7 | mars_7_e1_d30 | adaptive_gps_only | 1.000000 | 30 | 6.202257 | 6.995925 | 6.933492 | 0.932550 | 13.189604 |
| mars_7 | mars_7_e1_d30 | adaptive_gps_only | 2.000000 | 30 | 6.075737 | 6.863724 | 6.798770 | 0.942041 | 12.984446 |
| mars_7 | mars_7_e1_d60 | held_gnss | — | 60 | 11.755433 | 12.384933 | 11.388457 | 4.867200 | 10.978942 |
| mars_7 | mars_7_e1_d60 | constant_velocity_gnss | — | 60 | 14.515574 | 14.916770 | 12.403844 | 8.285813 | 16.957036 |
| mars_7 | mars_7_e1_d60 | damped_cv | — | 60 | 12.025496 | 12.620712 | 11.484489 | 5.233439 | 11.410759 |
| mars_7 | mars_7_e1_d60 | adaptive_full | 0.000000 | 60 | 14.342481 | 15.057232 | 13.435846 | 6.796933 | 14.544488 |
| mars_7 | mars_7_e1_d60 | adaptive_full | 1.000000 | 60 | 14.589585 | 15.224119 | 13.494906 | 7.047079 | 15.703158 |
| mars_7 | mars_7_e1_d60 | adaptive_full | 2.000000 | 60 | 15.898883 | 16.805434 | 16.197231 | 4.480213 | 17.215714 |
| mars_7 | mars_7_e1_d60 | adaptive_gps_only | 0.000000 | 60 | 17.121416 | 17.918366 | 17.270392 | 4.775078 | 20.174684 |
| mars_7 | mars_7_e1_d60 | adaptive_gps_only | 1.000000 | 60 | 16.979158 | 17.768590 | 17.082231 | 4.890826 | 19.893202 |
| mars_7 | mars_7_e1_d60 | adaptive_gps_only | 2.000000 | 60 | 16.794613 | 17.574474 | 16.829006 | 5.064257 | 19.462099 |
| mars_7 | mars_7_control | held_gnss | — | 0 | — | 1.984645 | 1.161668 | 1.609143 | — |
| mars_7 | mars_7_control | constant_velocity_gnss | — | 0 | — | 1.984645 | 1.161668 | 1.609143 | — |
| mars_7 | mars_7_control | damped_cv | — | 0 | — | 1.984645 | 1.161668 | 1.609143 | — |
| mars_7 | mars_7_control | adaptive_full | 0.000000 | 0 | — | 1.984645 | 1.161668 | 1.609143 | — |
| mars_7 | mars_7_control | adaptive_full | 1.000000 | 0 | — | 1.984645 | 1.161668 | 1.609143 | — |
| mars_7 | mars_7_control | adaptive_full | 2.000000 | 0 | — | 1.984645 | 1.161668 | 1.609143 | — |
| mars_7 | mars_7_control | adaptive_gps_only | 0.000000 | 0 | — | 1.984645 | 1.161668 | 1.609143 | — |
| mars_7 | mars_7_control | adaptive_gps_only | 1.000000 | 0 | — | 1.984645 | 1.161668 | 1.609143 | — |
| mars_7 | mars_7_control | adaptive_gps_only | 2.000000 | 0 | — | 1.984645 | 1.161668 | 1.609143 | — |

## Duration and flight summaries

Mean and sample SD across three seed-level episode means; SD is not a confidence interval.

| duration_s | method | mean_m | seed_sd_m | n_seeds |
|---|---|---|---|---|
| 10 | adaptive_full | 1.356255 | 0.096968 | 3 |
| 10 | adaptive_gps_only | 2.054597 | 0.048054 | 3 |
| 10 | constant_velocity_gnss | 0.706242 | — | 0 |
| 10 | damped_cv | 0.391840 | — | 0 |
| 10 | held_gnss | 0.142490 | — | 0 |
| 30 | adaptive_full | 8.972275 | 0.152832 | 3 |
| 30 | adaptive_gps_only | 9.907792 | 0.115160 | 3 |
| 30 | constant_velocity_gnss | 6.246879 | — | 0 |
| 30 | damped_cv | 6.218624 | — | 0 |
| 30 | held_gnss | 6.262051 | — | 0 |
| 60 | adaptive_full | 17.221968 | 0.925229 | 3 |
| 60 | adaptive_gps_only | 19.324801 | 0.197217 | 3 |
| 60 | constant_velocity_gnss | 14.392707 | — | 0 |
| 60 | damped_cv | 14.161105 | — | 0 |
| 60 | held_gnss | 14.214615 | — | 0 |

| flight | duration_s | method | mean_m | seed_sd_m | n_seeds |
|---|---|---|---|---|---|
| mars_6 | 10 | adaptive_full | 0.812568 | 0.431287 | 3 |
| mars_6 | 10 | adaptive_gps_only | 2.016123 | 0.048956 | 3 |
| mars_6 | 10 | constant_velocity_gnss | 0.643520 | — | 0 |
| mars_6 | 10 | damped_cv | 0.365843 | — | 0 |
| mars_6 | 10 | held_gnss | 0.225535 | — | 0 |
| mars_6 | 30 | adaptive_full | 12.979377 | 0.283949 | 3 |
| mars_6 | 30 | adaptive_gps_only | 13.618722 | 0.112194 | 3 |
| mars_6 | 30 | constant_velocity_gnss | 9.284420 | — | 0 |
| mars_6 | 30 | damped_cv | 10.454532 | — | 0 |
| mars_6 | 30 | held_gnss | 10.797290 | — | 0 |
| mars_6 | 60 | adaptive_full | 19.500286 | 1.075859 | 3 |
| mars_6 | 60 | adaptive_gps_only | 21.684541 | 0.231085 | 3 |
| mars_6 | 60 | constant_velocity_gnss | 14.269841 | — | 0 |
| mars_6 | 60 | damped_cv | 16.296715 | — | 0 |
| mars_6 | 60 | held_gnss | 16.673797 | — | 0 |
| mars_7 | 10 | adaptive_full | 1.899942 | 0.389656 | 3 |
| mars_7 | 10 | adaptive_gps_only | 2.093071 | 0.047153 | 3 |
| mars_7 | 10 | constant_velocity_gnss | 0.768964 | — | 0 |
| mars_7 | 10 | damped_cv | 0.417836 | — | 0 |
| mars_7 | 10 | held_gnss | 0.059445 | — | 0 |
| mars_7 | 30 | adaptive_full | 4.965172 | 0.288764 | 3 |
| mars_7 | 30 | adaptive_gps_only | 6.196863 | 0.118521 | 3 |
| mars_7 | 30 | constant_velocity_gnss | 3.209337 | — | 0 |
| mars_7 | 30 | damped_cv | 1.982717 | — | 0 |
| mars_7 | 30 | held_gnss | 1.726812 | — | 0 |
| mars_7 | 60 | adaptive_full | 14.943649 | 0.836432 | 3 |
| mars_7 | 60 | adaptive_gps_only | 16.965062 | 0.163857 | 3 |
| mars_7 | 60 | constant_velocity_gnss | 14.515574 | — | 0 |
| mars_7 | 60 | damped_cv | 12.025496 | — | 0 |
| mars_7 | 60 | held_gnss | 11.755433 | — | 0 |

## Matched sensor ablation

| seed | full_wins | outage_episodes |
|---|---|---|
| 0 | 5 | 6 |
| 1 | 6 | 6 |
| 2 | 6 | 6 |

| flight | scenario | duration_s | mean_difference_m | seed_sd_difference_m | full_wins_of_3 |
|---|---|---|---|---|---|
| mars_6 | mars_6_e1_d10 | 10 | -1.203555 | 0.470642 | 3 |
| mars_6 | mars_6_e1_d30 | 30 | -0.639345 | 0.297241 | 3 |
| mars_6 | mars_6_e1_d60 | 60 | -2.184254 | 1.259883 | 3 |
| mars_7 | mars_7_e1_d10 | 10 | -0.193129 | 0.343668 | 2 |
| mars_7 | mars_7_e1_d30 | 30 | -1.231691 | 0.206062 | 3 |
| mars_7 | mars_7_e1_d60 | 60 | -2.021413 | 0.994119 | 3 |

Full is lower than gps_only in 17/18 episode–seed pairs, with all-three-seed wins in 5/6 episodes. These are repeated trainings evaluated on two flights, not independent replicated test flights.

## Comparison with deterministic baselines by flight

| flight | duration_s | method | reference | mean_m | reference_mean_m | difference_m | difference_percent | wins_episode_seed | pairs |
|---|---|---|---|---|---|---|---|---|---|
| mars_6 | 10 | adaptive_full | held_gnss | 0.812568 | 0.225535 | 0.587033 | 260.284007 | 0 | 3 |
| mars_6 | 10 | adaptive_full | constant_velocity_gnss | 0.812568 | 0.643520 | 0.169048 | 26.269250 | 2 | 3 |
| mars_6 | 10 | adaptive_full | damped_cv | 0.812568 | 0.365843 | 0.446725 | 122.108308 | 0 | 3 |
| mars_6 | 30 | adaptive_full | held_gnss | 12.979377 | 10.797290 | 2.182087 | 20.209575 | 0 | 3 |
| mars_6 | 30 | adaptive_full | constant_velocity_gnss | 12.979377 | 9.284420 | 3.694957 | 39.797391 | 0 | 3 |
| mars_6 | 30 | adaptive_full | damped_cv | 12.979377 | 10.454532 | 2.524845 | 24.150723 | 0 | 3 |
| mars_6 | 60 | adaptive_full | held_gnss | 19.500286 | 16.673797 | 2.826489 | 16.951682 | 0 | 3 |
| mars_6 | 60 | adaptive_full | constant_velocity_gnss | 19.500286 | 14.269841 | 5.230445 | 36.653846 | 0 | 3 |
| mars_6 | 60 | adaptive_full | damped_cv | 19.500286 | 16.296715 | 3.203572 | 19.657777 | 0 | 3 |
| mars_7 | 10 | adaptive_full | held_gnss | 1.899942 | 0.059445 | 1.840497 | 3096.147249 | 0 | 3 |
| mars_7 | 10 | adaptive_full | constant_velocity_gnss | 1.899942 | 0.768964 | 1.130978 | 147.078087 | 0 | 3 |
| mars_7 | 10 | adaptive_full | damped_cv | 1.899942 | 0.417836 | 1.482105 | 354.709467 | 0 | 3 |
| mars_7 | 30 | adaptive_full | held_gnss | 4.965172 | 1.726812 | 3.238360 | 187.533990 | 0 | 3 |
| mars_7 | 30 | adaptive_full | constant_velocity_gnss | 4.965172 | 3.209337 | 1.755835 | 54.710196 | 0 | 3 |
| mars_7 | 30 | adaptive_full | damped_cv | 4.965172 | 1.982717 | 2.982456 | 150.422676 | 0 | 3 |
| mars_7 | 60 | adaptive_full | held_gnss | 14.943649 | 11.755433 | 3.188216 | 27.121212 | 0 | 3 |
| mars_7 | 60 | adaptive_full | constant_velocity_gnss | 14.943649 | 14.515574 | 0.428075 | 2.949077 | 1 | 3 |
| mars_7 | 60 | adaptive_full | damped_cv | 14.943649 | 12.025496 | 2.918153 | 24.266384 | 0 | 3 |

## Interpretation limits

All fixed episodes and seeds are retained, including unfavorable differences. A full/gps_only difference is distinct from an advantage over Held/CV/Damped-CV. Results support or limit claims only within these two held-out flights. No test-driven checkpoint selection, tuning or retraining occurred. No isolated gate effect, stop-detector behavior, INS/EKF superiority, statistical significance or generalization beyond these recordings is established. Exact scientific interpretation must respect the signs and magnitudes in the per-flight tables.
