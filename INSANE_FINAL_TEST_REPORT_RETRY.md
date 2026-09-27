# INSANE FINAL TEST REPORT

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

Status: **complete**. Один финальный test-only job: 48 model×scenario evaluations + 24 deterministic baseline×scenario evaluations, всего 72. Обучение/optimizer steps: 0; repeats: 0. Run ID и Kaggle version — в download/submission receipt. Test: mars_6/mars_7; validation в эти метрики не включена.

## 1. Все test-полёты, эпизоды и seeds

Primary — relative-motion RMSE3D внутри outage, метры. Baselines представлены один раз; seed пуст. Controls без искусственного outage сохраняются с N/A в outage-метриках. Ничего не исключено по ошибкам или движению.

| flight | scenario | method | seed | duration_s | n_native_gt | relative_motion_rmse3d_m | absolute_rmse3d_m | absolute_rmse_h_m | absolute_rmse_v_m | final_unavailable_error_m | final_relative_motion_error_m | recovery_first_5s_rmse3d_m | first_recovered_fix_error_m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mars_6 | mars_6_e1_d10 | held_gnss | — | 10 | 43 | 0.225535 | 5.030611 | 4.429654 | 2.384370 | 5.595697 | 0.949007 | 3.961722 | 4.041484 |
| mars_6 | mars_6_e1_d10 | constant_velocity_gnss | — | 10 | 43 | 0.643520 | 4.587604 | 4.205312 | 1.833430 | 4.575199 | 0.424102 | 3.907322 | 4.041484 |
| mars_6 | mars_6_e1_d10 | damped_cv | — | 10 | 43 | 0.365843 | 4.765937 | 4.298659 | 2.058078 | 5.138968 | 0.422388 | 3.936336 | 4.041484 |
| mars_6 | mars_6_e1_d10 | adaptive_full | 0.000000 | 10 | 43 | 0.639842 | 5.316423 | 4.698522 | 2.487618 | 6.556761 | 1.658350 | 4.021717 | 4.041484 |
| mars_6 | mars_6_e1_d10 | adaptive_full | 1.000000 | 10 | 43 | 0.494416 | 5.256856 | 4.711026 | 2.332544 | 6.466985 | 1.518256 | 4.016231 | 4.041484 |
| mars_6 | mars_6_e1_d10 | adaptive_full | 2.000000 | 10 | 43 | 1.303446 | 5.714213 | 5.421791 | 1.804554 | 6.930753 | 2.289058 | 4.045838 | 4.041484 |
| mars_6 | mars_6_e1_d10 | adaptive_gps_only | 0.000000 | 10 | 43 | 2.064416 | 6.446230 | 6.118648 | 2.028800 | 8.315769 | 3.688026 | 4.151016 | 4.041484 |
| mars_6 | mars_6_e1_d10 | adaptive_gps_only | 1.000000 | 10 | 43 | 2.017422 | 6.416073 | 6.076666 | 2.059155 | 8.259225 | 3.614865 | 4.146443 | 4.041484 |
| mars_6 | mars_6_e1_d10 | adaptive_gps_only | 2.000000 | 10 | 43 | 1.966530 | 6.371779 | 6.026497 | 2.069035 | 8.173285 | 3.524295 | 4.139500 | 4.041484 |
| mars_6 | mars_6_e1_d30 | held_gnss | — | 30 | 127 | 10.797290 | 14.227856 | 12.390379 | 6.993597 | 26.997701 | 23.141934 | 3.305089 | 3.303064 |
| mars_6 | mars_6_e1_d30 | constant_velocity_gnss | — | 30 | 127 | 9.284420 | 12.563074 | 11.527801 | 4.994061 | 24.266017 | 20.595118 | 3.305089 | 3.303064 |
| mars_6 | mars_6_e1_d30 | damped_cv | — | 30 | 127 | 10.454532 | 13.825744 | 12.193876 | 6.516179 | 26.518524 | 22.687661 | 3.305089 | 3.303064 |
| mars_6 | mars_6_e1_d30 | adaptive_full | 0.000000 | 30 | 127 | 13.204409 | 16.879409 | 15.869406 | 5.751211 | 31.723704 | 27.338545 | 3.305089 | 3.303064 |
| mars_6 | mars_6_e1_d30 | adaptive_full | 1.000000 | 30 | 127 | 12.660348 | 16.422288 | 15.219335 | 6.169552 | 30.440068 | 25.905941 | 3.305089 | 3.303064 |
| mars_6 | mars_6_e1_d30 | adaptive_full | 2.000000 | 30 | 127 | 13.073373 | 16.657671 | 16.008390 | 4.605370 | 31.151045 | 27.070111 | 3.305089 | 3.303064 |
| mars_6 | mars_6_e1_d30 | adaptive_gps_only | 0.000000 | 30 | 127 | 13.719452 | 17.637189 | 16.677976 | 5.737207 | 31.595606 | 27.026963 | 3.305089 | 3.303064 |
| mars_6 | mars_6_e1_d30 | adaptive_gps_only | 1.000000 | 30 | 127 | 13.638908 | 17.558929 | 16.558190 | 5.843145 | 31.473050 | 26.902661 | 3.305089 | 3.303064 |
| mars_6 | mars_6_e1_d30 | adaptive_gps_only | 2.000000 | 30 | 127 | 13.497806 | 17.416883 | 16.392282 | 5.885653 | 31.231575 | 26.663268 | 3.305089 | 3.303064 |
| mars_6 | mars_6_e1_d60 | held_gnss | — | 60 | 258 | 16.673797 | 19.972434 | 17.797601 | 9.063306 | 15.271053 | 10.814734 | 2.825150 | 2.794409 |
| mars_6 | mars_6_e1_d60 | constant_velocity_gnss | — | 60 | 258 | 14.269841 | 17.189137 | 16.290425 | 5.485299 | 11.146995 | 8.119251 | 2.825150 | 2.794409 |
| mars_6 | mars_6_e1_d60 | damped_cv | — | 60 | 258 | 16.296715 | 19.545920 | 17.584847 | 8.533239 | 14.815707 | 10.396928 | 2.825150 | 2.794409 |
| mars_6 | mars_6_e1_d60 | adaptive_full | 0.000000 | 60 | 258 | 19.305146 | 23.183607 | 22.020094 | 7.252247 | 20.575295 | 15.812785 | 2.825150 | 2.794409 |
| mars_6 | mars_6_e1_d60 | adaptive_full | 1.000000 | 60 | 258 | 18.535354 | 22.491135 | 21.110573 | 7.758536 | 20.177546 | 15.469911 | 2.825150 | 2.794409 |
| mars_6 | mars_6_e1_d60 | adaptive_full | 2.000000 | 60 | 258 | 20.660360 | 24.269370 | 23.746407 | 5.011036 | 26.414808 | 22.367412 | 2.825150 | 2.794409 |
| mars_6 | mars_6_e1_d60 | adaptive_gps_only | 0.000000 | 60 | 258 | 21.888443 | 25.990045 | 25.094804 | 6.762638 | 32.653690 | 28.277686 | 2.825150 | 2.794409 |
| mars_6 | mars_6_e1_d60 | adaptive_gps_only | 1.000000 | 60 | 258 | 21.731647 | 25.839739 | 24.889606 | 6.942593 | 32.310088 | 27.908232 | 2.825150 | 2.794409 |
| mars_6 | mars_6_e1_d60 | adaptive_gps_only | 2.000000 | 60 | 258 | 21.433532 | 25.543171 | 24.557210 | 7.028299 | 31.683549 | 27.268632 | 2.825150 | 2.794409 |
| mars_6 | mars_6_control | held_gnss | — | 0 | 392 | — | 4.053204 | 3.498657 | 2.046426 | — | — | — | — |
| mars_6 | mars_6_control | constant_velocity_gnss | — | 0 | 392 | — | 4.053204 | 3.498657 | 2.046426 | — | — | — | — |
| mars_6 | mars_6_control | damped_cv | — | 0 | 392 | — | 4.053204 | 3.498657 | 2.046426 | — | — | — | — |
| mars_6 | mars_6_control | adaptive_full | 0.000000 | 0 | 392 | — | 4.053204 | 3.498657 | 2.046426 | — | — | — | — |
| mars_6 | mars_6_control | adaptive_full | 1.000000 | 0 | 392 | — | 4.053204 | 3.498657 | 2.046426 | — | — | — | — |
| mars_6 | mars_6_control | adaptive_full | 2.000000 | 0 | 392 | — | 4.053204 | 3.498657 | 2.046426 | — | — | — | — |
| mars_6 | mars_6_control | adaptive_gps_only | 0.000000 | 0 | 392 | — | 4.053204 | 3.498657 | 2.046426 | — | — | — | — |
| mars_6 | mars_6_control | adaptive_gps_only | 1.000000 | 0 | 392 | — | 4.053204 | 3.498657 | 2.046426 | — | — | — | — |
| mars_6 | mars_6_control | adaptive_gps_only | 2.000000 | 0 | 392 | — | 4.053204 | 3.498657 | 2.046426 | — | — | — | — |
| mars_7 | mars_7_e1_d10 | held_gnss | — | 10 | 47 | 0.059445 | 1.521070 | 1.385516 | 0.627694 | 1.528813 | 0.308441 | 1.782027 | 1.596296 |
| mars_7 | mars_7_e1_d10 | constant_velocity_gnss | — | 10 | 47 | 0.768964 | 1.511978 | 1.477898 | 0.319214 | 1.837614 | 1.513682 | 1.800391 | 1.596296 |
| mars_7 | mars_7_e1_d10 | damped_cv | — | 10 | 47 | 0.417836 | 1.462931 | 1.424993 | 0.331002 | 1.551552 | 0.815529 | 1.783888 | 1.596296 |
| mars_7 | mars_7_e1_d10 | adaptive_full | 0.000000 | 10 | 47 | 2.229051 | 3.350971 | 3.246363 | 0.830742 | 4.221225 | 2.983247 | 2.012449 | 1.596296 |
| mars_7 | mars_7_e1_d10 | adaptive_full | 1.000000 | 10 | 47 | 2.001087 | 2.869968 | 2.823376 | 0.515042 | 3.413359 | 2.555071 | 1.922313 | 1.596296 |
| mars_7 | mars_7_e1_d10 | adaptive_full | 2.000000 | 10 | 47 | 1.469686 | 2.834093 | 2.685524 | 0.905562 | 3.766429 | 2.317129 | 1.961306 | 1.596296 |
| mars_7 | mars_7_e1_d10 | adaptive_gps_only | 0.000000 | 10 | 47 | 2.139733 | 3.037441 | 2.967149 | 0.649674 | 4.253047 | 3.384922 | 2.014640 | 1.596296 |
| mars_7 | mars_7_e1_d10 | adaptive_gps_only | 1.000000 | 10 | 47 | 2.094038 | 2.986741 | 2.920779 | 0.624236 | 4.169735 | 3.310982 | 2.004810 | 1.596296 |
| mars_7 | mars_7_e1_d10 | adaptive_gps_only | 2.000000 | 10 | 47 | 2.045442 | 2.936317 | 2.874937 | 0.597239 | 4.084101 | 3.231713 | 1.994867 | 1.596296 |
| mars_7 | mars_7_e1_d30 | held_gnss | — | 30 | 122 | 1.726812 | 2.742962 | 2.579837 | 0.931818 | 4.387730 | 4.216519 | 2.055122 | 1.960217 |
| mars_7 | mars_7_e1_d30 | constant_velocity_gnss | — | 30 | 122 | 3.209337 | 3.633219 | 2.999768 | 2.049798 | 7.246716 | 7.383134 | 2.368640 | 1.960217 |
| mars_7 | mars_7_e1_d30 | damped_cv | — | 30 | 122 | 1.982717 | 2.827831 | 2.665273 | 0.944959 | 4.780636 | 4.685937 | 2.091038 | 1.960217 |
| mars_7 | mars_7_e1_d30 | adaptive_full | 0.000000 | 30 | 122 | 5.085568 | 6.212921 | 6.093301 | 1.213289 | 8.792760 | 8.101682 | 2.575601 | 1.960217 |
| mars_7 | mars_7_e1_d30 | adaptive_full | 1.000000 | 30 | 122 | 5.174257 | 5.977551 | 5.854862 | 1.204872 | 9.511223 | 9.103501 | 2.679776 | 1.960217 |
| mars_7 | mars_7_e1_d30 | adaptive_full | 2.000000 | 30 | 122 | 4.635692 | 5.893001 | 5.769785 | 1.198769 | 8.919758 | 7.858300 | 2.596264 | 1.960217 |
| mars_7 | mars_7_e1_d30 | adaptive_gps_only | 0.000000 | 30 | 122 | 6.312595 | 7.113220 | 7.051573 | 0.934457 | 13.357061 | 12.801197 | 3.304734 | 1.960217 |
| mars_7 | mars_7_e1_d30 | adaptive_gps_only | 1.000000 | 30 | 122 | 6.202257 | 6.995925 | 6.933492 | 0.932550 | 13.189604 | 12.647568 | 3.276340 | 1.960217 |
| mars_7 | mars_7_e1_d30 | adaptive_gps_only | 2.000000 | 30 | 122 | 6.075737 | 6.863724 | 6.798770 | 0.942041 | 12.984446 | 12.458326 | 3.241648 | 1.960217 |
| mars_7 | mars_7_e1_d60 | held_gnss | — | 60 | 241 | 11.755433 | 12.384933 | 11.388457 | 4.867200 | 10.978942 | 10.441310 | 3.202091 | 2.578855 |
| mars_7 | mars_7_e1_d60 | constant_velocity_gnss | — | 60 | 241 | 14.515574 | 14.916770 | 12.403844 | 8.285813 | 16.957036 | 16.821486 | 4.023003 | 2.578855 |
| mars_7 | mars_7_e1_d60 | damped_cv | — | 60 | 241 | 12.025496 | 12.620712 | 11.484489 | 5.233439 | 11.410759 | 10.920958 | 3.254925 | 2.578855 |
| mars_7 | mars_7_e1_d60 | adaptive_full | 0.000000 | 60 | 241 | 14.342481 | 15.057232 | 13.435846 | 6.796933 | 14.544488 | 14.330708 | 3.674230 | 2.578855 |
| mars_7 | mars_7_e1_d60 | adaptive_full | 1.000000 | 60 | 241 | 14.589585 | 15.224119 | 13.494906 | 7.047079 | 15.703158 | 15.390169 | 3.840971 | 2.578855 |
| mars_7 | mars_7_e1_d60 | adaptive_full | 2.000000 | 60 | 241 | 15.898883 | 16.805434 | 16.197231 | 4.480213 | 17.215714 | 16.172273 | 4.067510 | 2.578855 |
| mars_7 | mars_7_e1_d60 | adaptive_gps_only | 0.000000 | 60 | 241 | 17.121416 | 17.918366 | 17.270392 | 4.775078 | 20.174684 | 19.224915 | 4.541783 | 2.578855 |
| mars_7 | mars_7_e1_d60 | adaptive_gps_only | 1.000000 | 60 | 241 | 16.979158 | 17.768590 | 17.082231 | 4.890826 | 19.893202 | 18.952461 | 4.496820 | 2.578855 |
| mars_7 | mars_7_e1_d60 | adaptive_gps_only | 2.000000 | 60 | 241 | 16.794613 | 17.574474 | 16.829006 | 5.064257 | 19.462099 | 18.536908 | 4.428194 | 2.578855 |
| mars_7 | mars_7_control | held_gnss | — | 0 | 393 | — | 1.984645 | 1.161668 | 1.609143 | — | — | — | — |
| mars_7 | mars_7_control | constant_velocity_gnss | — | 0 | 393 | — | 1.984645 | 1.161668 | 1.609143 | — | — | — | — |
| mars_7 | mars_7_control | damped_cv | — | 0 | 393 | — | 1.984645 | 1.161668 | 1.609143 | — | — | — | — |
| mars_7 | mars_7_control | adaptive_full | 0.000000 | 0 | 393 | — | 1.984645 | 1.161668 | 1.609143 | — | — | — | — |
| mars_7 | mars_7_control | adaptive_full | 1.000000 | 0 | 393 | — | 1.984645 | 1.161668 | 1.609143 | — | — | — | — |
| mars_7 | mars_7_control | adaptive_full | 2.000000 | 0 | 393 | — | 1.984645 | 1.161668 | 1.609143 | — | — | — | — |
| mars_7 | mars_7_control | adaptive_gps_only | 0.000000 | 0 | 393 | — | 1.984645 | 1.161668 | 1.609143 | — | — | — | — |
| mars_7 | mars_7_control | adaptive_gps_only | 1.000000 | 0 | 393 | — | 1.984645 | 1.161668 | 1.609143 | — | — | — | — |
| mars_7 | mars_7_control | adaptive_gps_only | 2.000000 | 0 | 393 | — | 1.984645 | 1.161668 | 1.609143 | — | — | — | — |

## 2. Агрегация

Сначала среднее episode RMSE каждой длительности внутри seed; затем mean и sample SD (ddof=1) трёх seed-средних. SD не является confidence interval. Baselines детерминированы: n_seeds=0, SD=N/A. Наложенные длительности и seeds не являются независимыми полётами.

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

| duration_s | method | seed | relative_motion_rmse3d_m |
|---|---|---|---|
| 10 | adaptive_full | 0.000000 | 1.434447 |
| 10 | adaptive_full | 1.000000 | 1.247751 |
| 10 | adaptive_full | 2.000000 | 1.386566 |
| 10 | adaptive_gps_only | 0.000000 | 2.102075 |
| 10 | adaptive_gps_only | 1.000000 | 2.055730 |
| 10 | adaptive_gps_only | 2.000000 | 2.005986 |
| 30 | adaptive_full | 0.000000 | 9.144989 |
| 30 | adaptive_full | 1.000000 | 8.917303 |
| 30 | adaptive_full | 2.000000 | 8.854532 |
| 30 | adaptive_gps_only | 0.000000 | 10.016023 |
| 30 | adaptive_gps_only | 1.000000 | 9.920583 |
| 30 | adaptive_gps_only | 2.000000 | 9.786771 |
| 60 | adaptive_full | 0.000000 | 16.823813 |
| 60 | adaptive_full | 1.000000 | 16.562469 |
| 60 | adaptive_full | 2.000000 | 18.279621 |
| 60 | adaptive_gps_only | 0.000000 | 19.504929 |
| 60 | adaptive_gps_only | 1.000000 | 19.355403 |
| 60 | adaptive_gps_only | 2.000000 | 19.114072 |

| flight | duration_s | method | seed | relative_motion_rmse3d_m |
|---|---|---|---|---|
| mars_6 | 10 | adaptive_full | 0.000000 | 0.639842 |
| mars_6 | 10 | adaptive_full | 1.000000 | 0.494416 |
| mars_6 | 10 | adaptive_full | 2.000000 | 1.303446 |
| mars_6 | 10 | adaptive_gps_only | 0.000000 | 2.064416 |
| mars_6 | 10 | adaptive_gps_only | 1.000000 | 2.017422 |
| mars_6 | 10 | adaptive_gps_only | 2.000000 | 1.966530 |
| mars_6 | 30 | adaptive_full | 0.000000 | 13.204409 |
| mars_6 | 30 | adaptive_full | 1.000000 | 12.660348 |
| mars_6 | 30 | adaptive_full | 2.000000 | 13.073373 |
| mars_6 | 30 | adaptive_gps_only | 0.000000 | 13.719452 |
| mars_6 | 30 | adaptive_gps_only | 1.000000 | 13.638908 |
| mars_6 | 30 | adaptive_gps_only | 2.000000 | 13.497806 |
| mars_6 | 60 | adaptive_full | 0.000000 | 19.305146 |
| mars_6 | 60 | adaptive_full | 1.000000 | 18.535354 |
| mars_6 | 60 | adaptive_full | 2.000000 | 20.660360 |
| mars_6 | 60 | adaptive_gps_only | 0.000000 | 21.888443 |
| mars_6 | 60 | adaptive_gps_only | 1.000000 | 21.731647 |
| mars_6 | 60 | adaptive_gps_only | 2.000000 | 21.433532 |
| mars_7 | 10 | adaptive_full | 0.000000 | 2.229051 |
| mars_7 | 10 | adaptive_full | 1.000000 | 2.001087 |
| mars_7 | 10 | adaptive_full | 2.000000 | 1.469686 |
| mars_7 | 10 | adaptive_gps_only | 0.000000 | 2.139733 |
| mars_7 | 10 | adaptive_gps_only | 1.000000 | 2.094038 |
| mars_7 | 10 | adaptive_gps_only | 2.000000 | 2.045442 |
| mars_7 | 30 | adaptive_full | 0.000000 | 5.085568 |
| mars_7 | 30 | adaptive_full | 1.000000 | 5.174257 |
| mars_7 | 30 | adaptive_full | 2.000000 | 4.635692 |
| mars_7 | 30 | adaptive_gps_only | 0.000000 | 6.312595 |
| mars_7 | 30 | adaptive_gps_only | 1.000000 | 6.202257 |
| mars_7 | 30 | adaptive_gps_only | 2.000000 | 6.075737 |
| mars_7 | 60 | adaptive_full | 0.000000 | 14.342481 |
| mars_7 | 60 | adaptive_full | 1.000000 | 14.589585 |
| mars_7 | 60 | adaptive_full | 2.000000 | 15.898883 |
| mars_7 | 60 | adaptive_gps_only | 0.000000 | 17.121416 |
| mars_7 | 60 | adaptive_gps_only | 1.000000 | 16.979158 |
| mars_7 | 60 | adaptive_gps_only | 2.000000 | 16.794613 |

## 3. A — full против matched gps_only

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

| flight | duration_s | method | reference | mean_m | reference_mean_m | difference_m | difference_percent | wins_episode_seed | pairs |
|---|---|---|---|---|---|---|---|---|---|
| mars_6 | 10 | adaptive_full | adaptive_gps_only | 0.812568 | 2.016123 | -1.203555 | -59.696498 | 3 | 3 |
| mars_6 | 30 | adaptive_full | adaptive_gps_only | 12.979377 | 13.618722 | -0.639345 | -4.694604 | 3 | 3 |
| mars_6 | 60 | adaptive_full | adaptive_gps_only | 19.500286 | 21.684541 | -2.184254 | -10.072863 | 3 | 3 |
| mars_7 | 10 | adaptive_full | adaptive_gps_only | 1.899942 | 2.093071 | -0.193129 | -9.227084 | 2 | 3 |
| mars_7 | 30 | adaptive_full | adaptive_gps_only | 4.965172 | 6.196863 | -1.231691 | -19.876035 | 3 | 3 |
| mars_7 | 60 | adaptive_full | adaptive_gps_only | 14.943649 | 16.965062 | -2.021413 | -11.915152 | 3 | 3 |

Full имеет меньшую ошибку в **17/18** episode×seed сравнениях; в **5/6** эпизодах выигрывает при всех трёх seeds. Знак full−gps_only показан для каждого test-полёта и длительности: отрицательный означает меньшую ошибку full. Эти counts описывают две отложенные записи, а не статистическое доказательство generalization.

## 4. B — сравнение с Held/CV/Damped-CV по полётам

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

Преимущество full над gps_only само по себе не означает превосходство над простыми baselines. Полётные таблицы имеют приоритет перед общим средним для этой интерпретации. Damped-CV τ=5 с зафиксировано ранее по train; здесь не подбиралось.

## 5. C — научные выводы и ограничения

Положительный matched contrast может поддержать пользу IMU в этих конкретных полётах; ухудшения, смена знака между seeds/полётами и выигрыши baselines ограничивают этот вывод. Результаты test не использованы для изменения модели, выбора seed, ensemble или новых параметров. Все шесть checkpoints выбраны ранее на validation. Два полёта одной кампании/платформы — ограниченный перенос, не доказательство общей применимости. Validation и test не объединяются.

Изолированный эффект gate не установлен: matched ungated control по трём seeds не проводился. g(t) — коэффициент управления прогнозом движения, не детектор остановки и не причинное объяснение результата. ESKF остаётся not_ready: отсутствуют документированное плечо ordinary GNSS и проверенная heading initialization; чисел ESKF нет. Нельзя заявлять превосходство над INS/EKF, доказанную новизну или готовность к Q2 на основании этих метрик.

## 6. Фиксация и проверки

`FINAL_EVALUATION_LOCK.json` содержит SHA256 шести best checkpoints, scalers/config, исходников, raw/calibration файлов и test manifest; он сохранён до inference. Scalers повторно не обучались. Mask-first GNSS features, causal IMU aggregation, actual dt, native GT support и разрешённая история остались прежними. В новом CSV split=test; старые CSV не изменялись.

Test manifest: 8 сценариев; unsupported: `[]`. Выбор: chronological onset stride60s, history20s/recovery10s, общие начала10/30/60s, прежние gap thresholds; fallback на поддерживаемые длительности только при отсутствии общего60s интервала, как в исходном алгоритме. Контроль — самый длинный непрерывный интервал каждого полёта. Нет padding наблюдений/склейки полётов.

Общие точки/маски, все восемь метрик из всех NPZ, paired deltas/percent, GT/GNSS native row provenance и g-интеграция независимо проверены. Веса в памяти и SHA256 файлов после inference совпали с зафиксированными. Verification: `{"status": "pass", "prediction_files_checked": 72, "summary_rows": 72, "paired_rows": 126, "duplicate_rows": 0, "max_metric_absolute_discrepancy_m": 3.552713678800501e-15, "max_paired_difference_discrepancy_m": 5.329070518200751e-15, "max_gate_integral_discrepancy_m": 5.338419751765855e-05, "all_points_and_masks_equal": true, "native_gt_and_gnss_provenance_verified_against_raw": true, "all_predictions_finite": true, "weights_unchanged": true, "model_inference_performed_by_verifier": false, "control_outage_metrics_are_na": true, "seed_count": 3, "independent_test_flights": 2}`.

Reference — опубликованная авторами dual-RTK/magnetometer позиция PX4 IMU. GT attitude не поступает в модель. Смещение ordinary GNSS antenna→IMU не документировано, поэтому остаётся ограничение геометрии/абсолютных ошибок. Существующие time corrections не применяются повторно. Primary использует один interpolated target anchor в момент последнего legal fix, но все оцениваемые GT timestamps исходные. Ни fitted alignment, ни GT reset нет.

## 7. Поведение gate и ресурсы

![All test gate traces](outputs/insane_final_test_retry/gate_traces.png)

| scenario | flight | duration_s | method | seed | g_mean | g_min | g_max | g_first | g_last |
|---|---|---|---|---|---|---|---|---|---|
| mars_6_e1_d10 | mars_6 | 10 | adaptive_full | 0 | 0.382310 | 0.351852 | 0.531353 | 0.375812 | 0.351852 |
| mars_6_e1_d10 | mars_6 | 10 | adaptive_full | 1 | 0.398726 | 0.378064 | 0.407887 | 0.402652 | 0.385126 |
| mars_6_e1_d10 | mars_6 | 10 | adaptive_full | 2 | 0.457051 | 0.441844 | 0.489312 | 0.441844 | 0.459288 |
| mars_6_e1_d10 | mars_6 | 10 | adaptive_gps_only | 0 | 0.309705 | 0.309331 | 0.311327 | 0.311327 | 0.309331 |
| mars_6_e1_d10 | mars_6 | 10 | adaptive_gps_only | 1 | 0.303189 | 0.302740 | 0.303737 | 0.303521 | 0.302740 |
| mars_6_e1_d10 | mars_6 | 10 | adaptive_gps_only | 2 | 0.321130 | 0.320571 | 0.322131 | 0.320974 | 0.320571 |
| mars_6_e1_d30 | mars_6 | 30 | adaptive_full | 0 | 0.421532 | 0.293425 | 0.570093 | 0.375812 | 0.412969 |
| mars_6_e1_d30 | mars_6 | 30 | adaptive_full | 1 | 0.392134 | 0.378064 | 0.407887 | 0.402652 | 0.383561 |
| mars_6_e1_d30 | mars_6 | 30 | adaptive_full | 2 | 0.460170 | 0.427367 | 0.489312 | 0.441844 | 0.461069 |
| mars_6_e1_d30 | mars_6 | 30 | adaptive_gps_only | 0 | 0.309040 | 0.308104 | 0.311327 | 0.311327 | 0.308104 |
| mars_6_e1_d30 | mars_6 | 30 | adaptive_gps_only | 1 | 0.302341 | 0.301126 | 0.303737 | 0.303521 | 0.301126 |
| mars_6_e1_d30 | mars_6 | 30 | adaptive_gps_only | 2 | 0.320090 | 0.318600 | 0.322131 | 0.320974 | 0.318600 |
| mars_6_e1_d60 | mars_6 | 60 | adaptive_full | 0 | 0.386286 | 0.293425 | 0.570093 | 0.375812 | 0.328314 |
| mars_6_e1_d60 | mars_6 | 60 | adaptive_full | 1 | 0.401917 | 0.378064 | 0.442367 | 0.402652 | 0.401528 |
| mars_6_e1_d60 | mars_6 | 60 | adaptive_full | 2 | 0.437833 | 0.377203 | 0.489312 | 0.441844 | 0.397381 |
| mars_6_e1_d60 | mars_6 | 60 | adaptive_gps_only | 0 | 0.308154 | 0.306408 | 0.311327 | 0.311327 | 0.306473 |
| mars_6_e1_d60 | mars_6 | 60 | adaptive_gps_only | 1 | 0.301212 | 0.299118 | 0.303737 | 0.303521 | 0.299128 |
| mars_6_e1_d60 | mars_6 | 60 | adaptive_gps_only | 2 | 0.318666 | 0.315863 | 0.322131 | 0.320974 | 0.315944 |
| mars_7_e1_d10 | mars_7 | 10 | adaptive_full | 0 | 0.455598 | 0.398816 | 0.620790 | 0.441844 | 0.587768 |
| mars_7_e1_d10 | mars_7 | 10 | adaptive_full | 1 | 0.393484 | 0.378947 | 0.427062 | 0.387167 | 0.405763 |
| mars_7_e1_d10 | mars_7 | 10 | adaptive_full | 2 | 0.476925 | 0.457307 | 0.494818 | 0.465491 | 0.491433 |
| mars_7_e1_d10 | mars_7 | 10 | adaptive_gps_only | 0 | 0.309598 | 0.309175 | 0.310935 | 0.310934 | 0.309233 |
| mars_7_e1_d10 | mars_7 | 10 | adaptive_gps_only | 1 | 0.303686 | 0.303222 | 0.304253 | 0.304078 | 0.303222 |
| mars_7_e1_d10 | mars_7 | 10 | adaptive_gps_only | 2 | 0.321780 | 0.321156 | 0.322780 | 0.321322 | 0.321238 |
| mars_7_e1_d30 | mars_7 | 30 | adaptive_full | 0 | 0.437684 | 0.297182 | 0.620790 | 0.441844 | 0.366040 |
| mars_7_e1_d30 | mars_7 | 30 | adaptive_full | 1 | 0.401132 | 0.373691 | 0.442148 | 0.387167 | 0.436958 |
| mars_7_e1_d30 | mars_7 | 30 | adaptive_full | 2 | 0.475509 | 0.420055 | 0.523649 | 0.465491 | 0.447697 |
| mars_7_e1_d30 | mars_7 | 30 | adaptive_gps_only | 0 | 0.308943 | 0.307902 | 0.310935 | 0.310934 | 0.308004 |
| mars_7_e1_d30 | mars_7 | 30 | adaptive_gps_only | 1 | 0.302847 | 0.301577 | 0.304253 | 0.304078 | 0.301577 |
| mars_7_e1_d30 | mars_7 | 30 | adaptive_gps_only | 2 | 0.320748 | 0.319162 | 0.322780 | 0.321322 | 0.319282 |
| mars_7_e1_d60 | mars_7 | 60 | adaptive_full | 0 | 0.407961 | 0.297182 | 0.620790 | 0.441844 | 0.432555 |
| mars_7_e1_d60 | mars_7 | 60 | adaptive_full | 1 | 0.418440 | 0.373691 | 0.471123 | 0.387167 | 0.446464 |
| mars_7_e1_d60 | mars_7 | 60 | adaptive_full | 2 | 0.456910 | 0.368037 | 0.523649 | 0.465491 | 0.409199 |
| mars_7_e1_d60 | mars_7 | 60 | adaptive_gps_only | 0 | 0.308049 | 0.306330 | 0.310935 | 0.310934 | 0.306370 |
| mars_7_e1_d60 | mars_7 | 60 | adaptive_gps_only | 1 | 0.301688 | 0.299535 | 0.304253 | 0.304078 | 0.299537 |
| mars_7_e1_d60 | mars_7 | 60 | adaptive_gps_only | 2 | 0.319325 | 0.316567 | 0.322780 | 0.321322 | 0.316610 |

Окружение: `{"python": "3.12.13", "torch": "2.10.0+cu128", "numpy": "2.0.2", "pandas": "2.3.3", "device": "cuda", "cuda": "12.8", "hardware": "Tesla T4"}`. Inference_ms — один synchronized forward/readout, не повторный latency benchmark. Held/CV разделяют время совместного расчёта. При возврате legal GNSS позиционный decoder использует legal fix, скрытое GRU-состояние сохраняется.

**STOP: финальный test завершён; повторного test или следующей версии автоматически не запускать.**
