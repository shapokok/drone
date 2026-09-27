# INSANE adaptive experiment

## Итог ограниченного эксперимента

Kaggle version 1; output run ID 353123708; six completed trainings. Local integrity verification: 99 prediction files, maximum saved-metric difference 7.11e-15 m. No model forward, training or test access during verification.

28 tests прошли до обучения. Выполнен один Kaggle job с шестью обучениями, без повторных запусков; test не загружался и не оценивался. Прежний pilot и его результаты сохранены отдельно.

**A — adaptive против простых baselines:** на mars_4 adaptive_full в среднем лучше Held, CV и Damped-CV при 10/30/60 с. На mars_5 сеть хуже Held и Damped-CV при всех трёх длительностях; против CV выигрывает на 30/60 с, а на 10 с средние почти совпадают. Универсальное превосходство не подтверждено. Damped-CV: τ=5 с, выбран по 540 train-эпизодам до validation-метрик. По средним нельзя заключать, что выигран каждый эпизод; все случаи приведены ниже.

**B — IMU против matched gps_only:** full выигрывает 24/27 сравнений эпизод×seed, по 8/9 при каждом seed. В 6/9 эпизодов выигрывает при всех трёх seeds. Средняя ошибка ниже у full на обоих validation-полётах при каждой длительности. Это воспроизводимая внутри данного протокола польза IMU, а не доказательство generalization на независимые полёты.

Основная метрика: среднее episode relative-motion RMSE3D внутри outage, м. ± — sample SD трёх seed-средних, не доверительный интервал. Процент — отношение разницы средних к среднему gps_only.

| Outage, с | Adaptive full, м | Adaptive gps_only, м | Full−gps_only, м | Разница, % | Held, м | CV, м | Damped-CV, м |
|---|---|---|---|---|---|---|---|
| 10 | 5.147 ± 0.104 | 6.340 ± 0.039 | -1.192 | -18.81 | 6.550 | 5.737 | 5.524 |
| 30 | 10.196 ± 0.479 | 12.735 ± 0.029 | -2.540 | -19.94 | 13.068 | 15.574 | 12.253 |
| 60 | 21.849 ± 1.730 | 24.694 ± 0.137 | -2.845 | -11.52 | 27.847 | 38.335 | 27.675 |

Gate показан для всех моделей/эпизодов в `gate_traces.png` и сохранён в NPZ. Среднее g по outage у full лежит примерно в 0.356–0.499, у gps_only — 0.300–0.326. Это наблюдение о коэффициенте прогноза, не детектор остановки и не причинное объяснение выигрыша. Gate обучался вместе с residual head; отдельного fixed-g контроля нет.

**Ограничение гипотезы:** прежний ungated pilot доступен только для seed0, поэтому из нового эксперимента нельзя выделить чистый эффект обучаемого gate относительно ungated-модели по трём seeds. Validation уже использовалась в предыдущем pilot и снова выбирает checkpoints. ESKF не готов и не оценивался. **STOP: test, новые seeds и следующая версия не запускались и автоматически не запускаются.**

Status **complete**. Six scheduled trainings, seeds 0/1/2; no test use. Kernel `shapok/drone-nav-insane-adaptive`; exact version/output ID in launch receipt. Code SHA256 `ea0480ffdb853952ef1184283b480646927be865c15e83eae71778abc644560c`.

## Two separate questions

A. Does adaptive prediction improve on Held, CV and train-selected Damped-CV? All flight/duration comparisons below use every fixed episode and all three seeds. Lower is better; improvements must be judged separately on each flight.

| flight | duration_s | method | reference | mean_rmse_m | reference_mean_rmse_m | delta_m | delta_percent | wins_episode_seed | episode_seed_pairs |
|---|---|---|---|---|---|---|---|---|---|
| mars_4 | 10 | adaptive_full | held_gnss | 7.253645 | 9.531708 | -2.278062 | -23.899836 | 3 | 6 |
| mars_4 | 10 | adaptive_full | constant_velocity_gnss | 7.253645 | 8.138682 | -0.885037 | -10.874447 | 4 | 6 |
| mars_4 | 10 | adaptive_full | damped_cv | 7.253645 | 8.153207 | -0.899562 | -11.033230 | 3 | 6 |
| mars_4 | 10 | adaptive_gps_only | held_gnss | 8.366421 | 9.531708 | -1.165287 | -12.225374 | 3 | 6 |
| mars_4 | 10 | adaptive_gps_only | constant_velocity_gnss | 8.366421 | 8.138682 | 0.227739 | 2.798228 | 3 | 6 |
| mars_4 | 10 | adaptive_gps_only | damped_cv | 8.366421 | 8.153207 | 0.213213 | 2.615087 | 3 | 6 |
| mars_4 | 30 | adaptive_full | held_gnss | 13.766577 | 19.301777 | -5.535199 | -28.677149 | 4 | 6 |
| mars_4 | 30 | adaptive_full | constant_velocity_gnss | 13.766577 | 21.617969 | -7.851391 | -36.318820 | 6 | 6 |
| mars_4 | 30 | adaptive_full | damped_cv | 13.766577 | 18.160710 | -4.394132 | -24.195818 | 4 | 6 |
| mars_4 | 30 | adaptive_gps_only | held_gnss | 15.893966 | 19.301777 | -3.407811 | -17.655425 | 3 | 6 |
| mars_4 | 30 | adaptive_gps_only | constant_velocity_gnss | 15.893966 | 21.617969 | -5.724003 | -26.477985 | 6 | 6 |
| mars_4 | 30 | adaptive_gps_only | damped_cv | 15.893966 | 18.160710 | -2.266744 | -12.481581 | 3 | 6 |
| mars_4 | 60 | adaptive_full | held_gnss | 27.520018 | 38.604564 | -11.084546 | -28.713045 | 6 | 6 |
| mars_4 | 60 | adaptive_full | constant_velocity_gnss | 27.520018 | 51.412846 | -23.892828 | -46.472487 | 6 | 6 |
| mars_4 | 60 | adaptive_full | damped_cv | 27.520018 | 38.132152 | -10.612134 | -27.829885 | 6 | 6 |
| mars_4 | 60 | adaptive_gps_only | held_gnss | 29.655989 | 38.604564 | -8.948575 | -23.180095 | 6 | 6 |
| mars_4 | 60 | adaptive_gps_only | constant_velocity_gnss | 29.655989 | 51.412846 | -21.756857 | -42.317940 | 6 | 6 |
| mars_4 | 60 | adaptive_gps_only | damped_cv | 29.655989 | 38.132152 | -8.476163 | -22.228389 | 6 | 6 |
| mars_5 | 10 | adaptive_full | held_gnss | 0.934966 | 0.587165 | 0.347801 | 59.233943 | 0 | 3 |
| mars_5 | 10 | adaptive_full | constant_velocity_gnss | 0.934966 | 0.933140 | 0.001826 | 0.195701 | 2 | 3 |
| mars_5 | 10 | adaptive_full | damped_cv | 0.934966 | 0.266359 | 0.668607 | 251.017559 | 0 | 3 |
| mars_5 | 10 | adaptive_gps_only | held_gnss | 2.286876 | 0.587165 | 1.699711 | 289.477659 | 0 | 3 |
| mars_5 | 10 | adaptive_gps_only | constant_velocity_gnss | 2.286876 | 0.933140 | 1.353737 | 145.073295 | 0 | 3 |
| mars_5 | 10 | adaptive_gps_only | damped_cv | 2.286876 | 0.266359 | 2.020518 | 758.570067 | 0 | 3 |
| mars_5 | 30 | adaptive_full | held_gnss | 3.054385 | 0.599547 | 2.454839 | 409.449183 | 0 | 3 |
| mars_5 | 30 | adaptive_full | constant_velocity_gnss | 3.054385 | 3.485750 | -0.431365 | -12.375098 | 3 | 3 |
| mars_5 | 30 | adaptive_full | damped_cv | 3.054385 | 0.436568 | 2.617817 | 599.635534 | 0 | 3 |
| mars_5 | 30 | adaptive_gps_only | held_gnss | 6.418177 | 0.599547 | 5.818631 | 970.505204 | 0 | 3 |
| mars_5 | 30 | adaptive_gps_only | constant_velocity_gnss | 6.418177 | 3.485750 | 2.932427 | 84.126145 | 0 | 3 |
| mars_5 | 30 | adaptive_gps_only | damped_cv | 6.418177 | 0.436568 | 5.981609 | 1370.143649 | 0 | 3 |
| mars_5 | 60 | adaptive_full | held_gnss | 10.507714 | 6.330691 | 4.177024 | 65.980535 | 0 | 3 |
| mars_5 | 60 | adaptive_full | constant_velocity_gnss | 10.507714 | 12.178496 | -1.670781 | -13.719111 | 3 | 3 |
| mars_5 | 60 | adaptive_full | damped_cv | 10.507714 | 6.760384 | 3.747330 | 55.430726 | 0 | 3 |
| mars_5 | 60 | adaptive_gps_only | held_gnss | 14.770553 | 6.330691 | 8.439862 | 133.316607 | 0 | 3 |
| mars_5 | 60 | adaptive_gps_only | constant_velocity_gnss | 14.770553 | 12.178496 | 2.592057 | 21.283887 | 0 | 3 |
| mars_5 | 60 | adaptive_gps_only | damped_cv | 14.770553 | 6.760384 | 8.010169 | 118.486882 | 0 | 3 |

B. Does IMU improve matched gps_only?

| flight | duration_s | method | reference | mean_rmse_m | reference_mean_rmse_m | delta_m | delta_percent | wins_episode_seed | episode_seed_pairs |
|---|---|---|---|---|---|---|---|---|---|
| mars_4 | 10 | adaptive_full | adaptive_gps_only | 7.253645 | 8.366421 | -1.112776 | -13.300497 | 5 | 6 |
| mars_4 | 30 | adaptive_full | adaptive_gps_only | 13.766577 | 15.893966 | -2.127389 | -13.384882 | 5 | 6 |
| mars_4 | 60 | adaptive_full | adaptive_gps_only | 27.520018 | 29.655989 | -2.135971 | -7.202495 | 5 | 6 |
| mars_5 | 10 | adaptive_full | adaptive_gps_only | 0.934966 | 2.286876 | -1.351910 | -59.116026 | 3 | 3 |
| mars_5 | 30 | adaptive_full | adaptive_gps_only | 3.054385 | 6.418177 | -3.363792 | -52.410396 | 3 | 3 |
| mars_5 | 60 | adaptive_full | adaptive_gps_only | 10.507714 | 14.770553 | -4.262839 | -28.860386 | 3 | 3 |

| seed | full_wins | outage_episodes |
|---|---|---|
| 0 | 8 | 9 |
| 1 | 8 | 9 |
| 2 | 8 | 9 |

| flight | scenario | duration_s | mean_delta_m | seed_sd_delta_m | wins_of_3 |
|---|---|---|---|---|---|
| mars_4 | mars_4_e1_d10 | 10 | -0.329438 | 0.488956 | 2 |
| mars_4 | mars_4_e1_d30 | 30 | -0.711266 | 1.282974 | 2 |
| mars_4 | mars_4_e1_d60 | 60 | -1.334316 | 1.082200 | 3 |
| mars_4 | mars_4_e2_d10 | 10 | -1.896113 | 0.805857 | 3 |
| mars_4 | mars_4_e2_d30 | 30 | -3.543511 | 2.393496 | 3 |
| mars_4 | mars_4_e2_d60 | 60 | -2.937626 | 3.715416 | 2 |
| mars_5 | mars_5_e1_d10 | 10 | -1.351910 | 0.105291 | 3 |
| mars_5 | mars_5_e1_d30 | 30 | -3.363792 | 0.153616 | 3 |
| mars_5 | mars_5_e1_d60 | 60 | -4.262839 | 0.731316 | 3 |

Adaptive full has lower error in 24/27 episode–seed pairs. All-three-seed wins occur in 6/9 fixed episodes. Three seeds are repeat trainings on the same two validation flights, not independent flights or evidence of generalization. No practical-effect threshold or significance claim is invented after results.

## Every episode and seed, before aggregate tables

Deterministic baselines appear once per episode; their seed is blank. Controls are retained, primary outage metric is undefined for them.

| flight | scenario | method | seed | duration_s | n_native_gt | relative_motion_rmse3d_m | absolute_rmse3d_m | absolute_rmse_h_m | absolute_rmse_v_m | final_unavailable_error_m | recovery_first_5s_rmse3d_m |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mars_4 | mars_4_e1_d10 | held_gnss | — | 10 | 29 | 0.505639 | 2.693868 | 1.993519 | 1.811851 | 3.000196 | 2.942734 |
| mars_4 | mars_4_e1_d10 | constant_velocity_gnss | — | 10 | 29 | 1.488202 | 2.454870 | 2.374678 | 0.622328 | 2.803890 | 2.932554 |
| mars_4 | mars_4_e1_d10 | damped_cv | — | 10 | 29 | 0.643494 | 2.378308 | 2.200172 | 0.903101 | 2.507746 | 2.916836 |
| mars_4 | mars_4_e1_d10 | adaptive_full | 0.000000 | 10 | 29 | 2.245816 | 1.812464 | 0.968375 | 1.532083 | 2.378102 | 2.911995 |
| mars_4 | mars_4_e1_d10 | adaptive_gps_only | 0.000000 | 10 | 29 | 2.245649 | 1.868625 | 1.100809 | 1.509960 | 2.277807 | 2.906933 |
| mars_4 | mars_4_e1_d10 | adaptive_full | 1.000000 | 10 | 29 | 2.104687 | 1.666779 | 1.079932 | 1.269606 | 1.811763 | 2.888242 |
| mars_4 | mars_4_e1_d10 | adaptive_gps_only | 1.000000 | 10 | 29 | 2.201941 | 1.844501 | 1.087010 | 1.490165 | 2.205216 | 2.903725 |
| mars_4 | mars_4_e1_d10 | adaptive_full | 2.000000 | 10 | 29 | 1.257889 | 1.961958 | 1.138108 | 1.598121 | 2.218375 | 2.904579 |
| mars_4 | mars_4_e1_d10 | adaptive_gps_only | 2.000000 | 10 | 29 | 2.149116 | 1.797118 | 1.071424 | 1.442804 | 2.086010 | 2.898702 |
| mars_4 | mars_4_e1_d30 | held_gnss | — | 30 | 115 | 5.346875 | 6.005079 | 5.283607 | 2.853853 | 17.189280 | 2.598311 |
| mars_4 | mars_4_e1_d30 | constant_velocity_gnss | — | 30 | 115 | 9.552351 | 9.454672 | 6.364851 | 6.991387 | 23.816958 | 2.598311 |
| mars_4 | mars_4_e1_d30 | damped_cv | — | 30 | 115 | 6.031368 | 6.431815 | 5.516607 | 3.306855 | 18.141667 | 2.598311 |
| mars_4 | mars_4_e1_d30 | adaptive_full | 0.000000 | 30 | 115 | 7.042583 | 5.566868 | 4.407670 | 3.400363 | 13.873318 | 2.598311 |
| mars_4 | mars_4_e1_d30 | adaptive_gps_only | 0.000000 | 30 | 115 | 7.338586 | 6.025753 | 5.000682 | 3.361975 | 15.276828 | 2.598311 |
| mars_4 | mars_4_e1_d30 | adaptive_full | 1.000000 | 30 | 115 | 7.611108 | 6.310663 | 4.967841 | 3.891661 | 16.388413 | 2.598311 |
| mars_4 | mars_4_e1_d30 | adaptive_gps_only | 1.000000 | 30 | 115 | 7.298466 | 6.001296 | 4.946365 | 3.398387 | 15.358209 | 2.598311 |
| mars_4 | mars_4_e1_d30 | adaptive_full | 2.000000 | 30 | 115 | 5.142671 | 4.203058 | 2.774030 | 3.157603 | 12.423926 | 2.598311 |
| mars_4 | mars_4_e1_d30 | adaptive_gps_only | 2.000000 | 30 | 115 | 7.293107 | 6.009302 | 4.882164 | 3.503740 | 15.564555 | 2.598311 |
| mars_4 | mars_4_e1_d60 | held_gnss | — | 60 | 231 | 24.967087 | 25.865798 | 24.548413 | 8.149533 | 49.253469 | 1.681252 |
| mars_4 | mars_4_e1_d60 | constant_velocity_gnss | — | 60 | 231 | 31.544700 | 31.992685 | 26.578197 | 17.808182 | 57.935246 | 1.681252 |
| mars_4 | mars_4_e1_d60 | damped_cv | — | 60 | 231 | 25.608786 | 26.435763 | 24.778595 | 9.212533 | 49.798587 | 1.681252 |
| mars_4 | mars_4_e1_d60 | adaptive_full | 0.000000 | 60 | 231 | 15.448980 | 15.141415 | 11.744442 | 9.556701 | 27.202696 | 1.681252 |
| mars_4 | mars_4_e1_d60 | adaptive_gps_only | 0.000000 | 60 | 231 | 17.767805 | 17.754141 | 14.773942 | 9.845819 | 31.243769 | 1.681252 |
| mars_4 | mars_4_e1_d60 | adaptive_full | 1.000000 | 60 | 231 | 17.745976 | 17.589502 | 13.827831 | 10.871138 | 27.963664 | 1.681252 |
| mars_4 | mars_4_e1_d60 | adaptive_gps_only | 1.000000 | 60 | 231 | 17.921543 | 17.921696 | 14.914904 | 9.936440 | 31.532986 | 1.681252 |
| mars_4 | mars_4_e1_d60 | adaptive_full | 2.000000 | 60 | 231 | 16.759504 | 16.954568 | 14.319291 | 9.078285 | 32.735688 | 1.681252 |
| mars_4 | mars_4_e1_d60 | adaptive_gps_only | 2.000000 | 60 | 231 | 18.268059 | 18.277205 | 15.144598 | 10.232173 | 32.124751 | 1.681252 |
| mars_4 | mars_4_e2_d10 | held_gnss | — | 10 | 48 | 18.557776 | 18.686558 | 18.527404 | 2.433668 | 25.511162 | 3.062152 |
| mars_4 | mars_4_e2_d10 | constant_velocity_gnss | — | 10 | 48 | 14.789161 | 15.603278 | 15.007244 | 4.271407 | 20.162089 | 3.062152 |
| mars_4 | mars_4_e2_d10 | damped_cv | — | 10 | 48 | 15.662921 | 16.153447 | 15.796709 | 3.376068 | 21.579321 | 3.062152 |
| mars_4 | mars_4_e2_d10 | adaptive_full | 0.000000 | 10 | 48 | 11.949581 | 12.159272 | 11.997450 | 1.977142 | 15.345884 | 3.062152 |
| mars_4 | mars_4_e2_d10 | adaptive_gps_only | 0.000000 | 10 | 48 | 14.509214 | 14.759723 | 14.498682 | 2.763631 | 19.290344 | 3.062152 |
| mars_4 | mars_4_e2_d10 | adaptive_full | 1.000000 | 10 | 48 | 12.461101 | 12.755575 | 12.471331 | 2.677793 | 16.154227 | 3.062152 |
| mars_4 | mars_4_e2_d10 | adaptive_gps_only | 1.000000 | 10 | 48 | 14.590443 | 14.837346 | 14.573467 | 2.785836 | 19.415947 | 3.062152 |
| mars_4 | mars_4_e2_d10 | adaptive_full | 2.000000 | 10 | 48 | 13.502795 | 13.817653 | 13.633613 | 2.247698 | 17.736736 | 3.062152 |
| mars_4 | mars_4_e2_d10 | adaptive_gps_only | 2.000000 | 10 | 48 | 14.502160 | 14.769271 | 14.495068 | 2.832732 | 19.278811 | 3.062152 |
| mars_4 | mars_4_e2_d30 | held_gnss | — | 30 | 120 | 33.256678 | 33.488176 | 33.362416 | 2.899504 | 58.065121 | 3.250258 |
| mars_4 | mars_4_e2_d30 | constant_velocity_gnss | — | 30 | 120 | 33.683586 | 34.617940 | 33.790171 | 7.525032 | 68.974943 | 3.250258 |
| mars_4 | mars_4_e2_d30 | damped_cv | — | 30 | 120 | 30.290051 | 30.771861 | 30.498100 | 4.095526 | 56.522212 | 3.250258 |
| mars_4 | mars_4_e2_d30 | adaptive_full | 0.000000 | 30 | 120 | 18.726778 | 19.169233 | 19.071389 | 1.934322 | 35.974318 | 3.250258 |
| mars_4 | mars_4_e2_d30 | adaptive_gps_only | 0.000000 | 30 | 120 | 24.379023 | 24.817848 | 24.538872 | 3.710704 | 45.091225 | 3.250258 |
| mars_4 | mars_4_e2_d30 | adaptive_full | 1.000000 | 30 | 120 | 20.488414 | 21.020631 | 20.714464 | 3.574623 | 39.899315 | 3.250258 |
| mars_4 | mars_4_e2_d30 | adaptive_gps_only | 1.000000 | 30 | 120 | 24.524735 | 24.957458 | 24.672312 | 3.761882 | 45.255012 | 3.250258 |
| mars_4 | mars_4_e2_d30 | adaptive_full | 2.000000 | 30 | 120 | 23.587910 | 24.121031 | 23.984918 | 2.558879 | 45.761894 | 3.250258 |
| mars_4 | mars_4_e2_d30 | adaptive_gps_only | 2.000000 | 30 | 120 | 24.529878 | 24.989457 | 24.685891 | 3.883265 | 45.625697 | 3.250258 |
| mars_4 | mars_4_e2_d60 | held_gnss | — | 60 | 244 | 52.242040 | 52.696970 | 52.429473 | 5.302927 | 30.689360 | 2.984149 |
| mars_4 | mars_4_e2_d60 | constant_velocity_gnss | — | 60 | 244 | 71.280993 | 72.290811 | 71.256854 | 12.182865 | 106.563506 | 2.984149 |
| mars_4 | mars_4_e2_d60 | damped_cv | — | 60 | 244 | 50.655518 | 51.274094 | 50.912657 | 6.077335 | 30.561323 | 2.984149 |
| mars_4 | mars_4_e2_d60 | adaptive_full | 0.000000 | 60 | 244 | 34.615932 | 35.357953 | 35.043327 | 4.706382 | 39.672511 | 2.984149 |
| mars_4 | mars_4_e2_d60 | adaptive_gps_only | 0.000000 | 60 | 244 | 41.106925 | 41.831230 | 41.392943 | 6.039548 | 30.677209 | 2.984149 |
| mars_4 | mars_4_e2_d60 | adaptive_full | 1.000000 | 60 | 244 | 37.924477 | 38.727875 | 38.281001 | 5.866283 | 41.517805 | 2.984149 |
| mars_4 | mars_4_e2_d60 | adaptive_gps_only | 1.000000 | 60 | 244 | 41.167366 | 41.885585 | 41.439665 | 6.095602 | 29.920884 | 2.984149 |
| mars_4 | mars_4_e2_d60 | adaptive_full | 2.000000 | 60 | 244 | 42.625239 | 43.422145 | 43.129452 | 5.033191 | 41.120424 | 2.984149 |
| mars_4 | mars_4_e2_d60 | adaptive_gps_only | 2.000000 | 60 | 244 | 41.704235 | 42.445308 | 41.983675 | 6.243013 | 32.313437 | 2.984149 |
| mars_4 | mars_4_control | held_gnss | — | 0 | 711 | — | 2.944914 | 1.530340 | 2.516064 | — | — |
| mars_4 | mars_4_control | constant_velocity_gnss | — | 0 | 711 | — | 2.944914 | 1.530340 | 2.516064 | — | — |
| mars_4 | mars_4_control | damped_cv | — | 0 | 711 | — | 2.944914 | 1.530340 | 2.516064 | — | — |
| mars_4 | mars_4_control | adaptive_full | 0.000000 | 0 | 711 | — | 2.944914 | 1.530340 | 2.516064 | — | — |
| mars_4 | mars_4_control | adaptive_gps_only | 0.000000 | 0 | 711 | — | 2.944914 | 1.530340 | 2.516064 | — | — |
| mars_4 | mars_4_control | adaptive_full | 1.000000 | 0 | 711 | — | 2.944914 | 1.530340 | 2.516064 | — | — |
| mars_4 | mars_4_control | adaptive_gps_only | 1.000000 | 0 | 711 | — | 2.944914 | 1.530340 | 2.516064 | — | — |
| mars_4 | mars_4_control | adaptive_full | 2.000000 | 0 | 711 | — | 2.944914 | 1.530340 | 2.516064 | — | — |
| mars_4 | mars_4_control | adaptive_gps_only | 2.000000 | 0 | 711 | — | 2.944914 | 1.530340 | 2.516064 | — | — |
| mars_5 | mars_5_e1_d10 | held_gnss | — | 10 | 40 | 0.587165 | 6.203632 | 3.596204 | 5.054935 | 6.220601 | 5.608907 |
| mars_5 | mars_5_e1_d10 | constant_velocity_gnss | — | 10 | 40 | 0.933140 | 5.185223 | 3.452228 | 3.868936 | 4.382373 | 5.515861 |
| mars_5 | mars_5_e1_d10 | damped_cv | — | 10 | 40 | 0.266359 | 5.622281 | 3.517288 | 4.386197 | 5.391488 | 5.563399 |
| mars_5 | mars_5_e1_d10 | adaptive_full | 0.000000 | 10 | 40 | 0.885320 | 5.479895 | 3.677688 | 4.062495 | 5.033316 | 5.545426 |
| mars_5 | mars_5_e1_d10 | adaptive_gps_only | 0.000000 | 10 | 40 | 2.339124 | 6.952039 | 4.948160 | 4.883294 | 7.776024 | 5.711799 |
| mars_5 | mars_5_e1_d10 | adaptive_full | 1.000000 | 10 | 40 | 1.044723 | 5.211417 | 3.121224 | 4.173347 | 4.462687 | 5.519382 |
| mars_5 | mars_5_e1_d10 | adaptive_gps_only | 1.000000 | 10 | 40 | 2.288245 | 6.916928 | 4.920792 | 4.861039 | 7.709771 | 5.706990 |
| mars_5 | mars_5_e1_d10 | adaptive_full | 2.000000 | 10 | 40 | 0.874855 | 5.730808 | 3.216467 | 4.743048 | 5.516741 | 5.569735 |
| mars_5 | mars_5_e1_d10 | adaptive_gps_only | 2.000000 | 10 | 40 | 2.233260 | 6.865781 | 4.889263 | 4.820173 | 7.616688 | 5.700302 |
| mars_5 | mars_5_e1_d30 | held_gnss | — | 30 | 122 | 0.599547 | 6.216036 | 3.594904 | 5.071072 | 6.223684 | 6.377110 |
| mars_5 | mars_5_e1_d30 | constant_velocity_gnss | — | 30 | 122 | 3.485750 | 4.052330 | 3.200578 | 2.485492 | 3.243321 | 6.217585 |
| mars_5 | mars_5_e1_d30 | damped_cv | — | 30 | 122 | 0.436568 | 5.405372 | 3.484370 | 4.132459 | 5.272612 | 6.315591 |
| mars_5 | mars_5_e1_d30 | adaptive_full | 0.000000 | 30 | 122 | 3.161060 | 5.021282 | 4.241681 | 2.687271 | 5.113456 | 6.306960 |
| mars_5 | mars_5_e1_d30 | adaptive_gps_only | 0.000000 | 30 | 122 | 6.549073 | 9.716546 | 8.562778 | 4.592394 | 13.618664 | 7.157409 |
| mars_5 | mars_5_e1_d30 | adaptive_full | 1.000000 | 30 | 122 | 3.223660 | 4.014996 | 2.772443 | 2.904093 | 2.683751 | 6.198162 |
| mars_5 | mars_5_e1_d30 | adaptive_gps_only | 1.000000 | 30 | 122 | 6.423164 | 9.597151 | 8.457761 | 4.535592 | 13.410116 | 7.129042 |
| mars_5 | mars_5_e1_d30 | adaptive_full | 2.000000 | 30 | 122 | 2.778436 | 5.700182 | 3.751573 | 4.291593 | 6.271070 | 6.381367 |
| mars_5 | mars_5_e1_d30 | adaptive_gps_only | 2.000000 | 30 | 122 | 6.282295 | 9.435066 | 8.335180 | 4.421000 | 13.127169 | 7.091074 |
| mars_5 | mars_5_e1_d60 | held_gnss | — | 60 | 226 | 6.330691 | 6.426742 | 4.980892 | 4.061248 | 2.973026 | 7.593655 |
| mars_5 | mars_5_e1_d60 | constant_velocity_gnss | — | 60 | 226 | 12.178496 | 9.214092 | 4.953948 | 7.769034 | 13.750179 | 8.302607 |
| mars_5 | mars_5_e1_d60 | damped_cv | — | 60 | 226 | 6.760384 | 6.077402 | 4.935935 | 3.545611 | 3.174200 | 7.598516 |
| mars_5 | mars_5_e1_d60 | adaptive_full | 0.000000 | 60 | 226 | 10.104763 | 7.955513 | 4.935554 | 6.239431 | 10.431197 | 7.997328 |
| mars_5 | mars_5_e1_d60 | adaptive_gps_only | 0.000000 | 60 | 226 | 14.951256 | 15.809037 | 15.306942 | 3.952616 | 19.612541 | 9.024566 |
| mars_5 | mars_5_e1_d60 | adaptive_full | 1.000000 | 60 | 226 | 10.277410 | 7.485751 | 5.149601 | 5.433053 | 9.257153 | 7.901782 |
| mars_5 | mars_5_e1_d60 | adaptive_gps_only | 1.000000 | 60 | 226 | 14.776942 | 15.606705 | 15.098342 | 3.950858 | 19.281690 | 8.979627 |
| mars_5 | mars_5_e1_d60 | adaptive_full | 2.000000 | 60 | 226 | 11.140970 | 10.180387 | 9.412512 | 3.878775 | 10.461633 | 7.997641 |
| mars_5 | mars_5_e1_d60 | adaptive_gps_only | 2.000000 | 60 | 226 | 14.583461 | 15.342690 | 14.816873 | 3.982262 | 18.815690 | 8.917197 |
| mars_5 | mars_5_control | held_gnss | — | 0 | 348 | — | 6.648175 | 3.486030 | 5.660904 | — | — |
| mars_5 | mars_5_control | constant_velocity_gnss | — | 0 | 348 | — | 6.648175 | 3.486030 | 5.660904 | — | — |
| mars_5 | mars_5_control | damped_cv | — | 0 | 348 | — | 6.648175 | 3.486030 | 5.660904 | — | — |
| mars_5 | mars_5_control | adaptive_full | 0.000000 | 0 | 348 | — | 6.648175 | 3.486030 | 5.660904 | — | — |
| mars_5 | mars_5_control | adaptive_gps_only | 0.000000 | 0 | 348 | — | 6.648175 | 3.486030 | 5.660904 | — | — |
| mars_5 | mars_5_control | adaptive_full | 1.000000 | 0 | 348 | — | 6.648175 | 3.486030 | 5.660904 | — | — |
| mars_5 | mars_5_control | adaptive_gps_only | 1.000000 | 0 | 348 | — | 6.648175 | 3.486030 | 5.660904 | — | — |
| mars_5 | mars_5_control | adaptive_full | 2.000000 | 0 | 348 | — | 6.648175 | 3.486030 | 5.660904 | — | — |
| mars_5 | mars_5_control | adaptive_gps_only | 2.000000 | 0 | 348 | — | 6.648175 | 3.486030 | 5.660904 | — | — |

## Aggregates

First average episode RMSE within a seed, then report mean and sample SD (ddof=1) of the three seed means. SD is not a confidence interval. Episode durations are nested, episodes can overlap. Overall duration means weight three episodes equally (two mars_4, one mars_5); flight tables make that imbalance explicit. Baselines have no seed SD.

| duration_s | method | mean_m | seed_sd_m | n_seeds |
|---|---|---|---|---|
| 10 | adaptive_full | 5.147419 | 0.104451 | 3 |
| 10 | adaptive_gps_only | 6.339906 | 0.039087 | 3 |
| 10 | constant_velocity_gnss | 5.736834 | — | 0 |
| 10 | damped_cv | 5.524258 | — | 0 |
| 10 | held_gnss | 6.550193 | — | 0 |
| 30 | adaptive_full | 10.195847 | 0.479371 | 3 |
| 30 | adaptive_gps_only | 12.735370 | 0.029303 | 3 |
| 30 | constant_velocity_gnss | 15.573896 | — | 0 |
| 30 | damped_cv | 12.252662 | — | 0 |
| 30 | held_gnss | 13.067700 | — | 0 |
| 60 | adaptive_full | 21.849250 | 1.729867 | 3 |
| 60 | adaptive_gps_only | 24.694177 | 0.136770 | 3 |
| 60 | constant_velocity_gnss | 38.334729 | — | 0 |
| 60 | damped_cv | 27.674896 | — | 0 |
| 60 | held_gnss | 27.846606 | — | 0 |

| flight | duration_s | method | mean_m | seed_sd_m | n_seeds |
|---|---|---|---|---|---|
| mars_4 | 10 | adaptive_full | 7.253645 | 0.143574 | 3 |
| mars_4 | 10 | adaptive_gps_only | 8.366421 | 0.036543 | 3 |
| mars_4 | 10 | constant_velocity_gnss | 8.138682 | — | 0 |
| mars_4 | 10 | damped_cv | 8.153207 | — | 0 |
| mars_4 | 10 | held_gnss | 9.531708 | — | 0 |
| mars_4 | 30 | adaptive_full | 13.766577 | 0.779870 | 3 |
| mars_4 | 30 | adaptive_gps_only | 15.893966 | 0.030451 | 3 |
| mars_4 | 30 | constant_velocity_gnss | 21.617969 | — | 0 |
| mars_4 | 30 | damped_cv | 18.160710 | — | 0 |
| mars_4 | 30 | held_gnss | 19.301777 | — | 0 |
| mars_4 | 60 | adaptive_full | 27.520018 | 2.345894 | 3 |
| mars_4 | 60 | adaptive_gps_only | 29.655989 | 0.290896 | 3 |
| mars_4 | 60 | constant_velocity_gnss | 51.412846 | — | 0 |
| mars_4 | 60 | damped_cv | 38.132152 | — | 0 |
| mars_4 | 60 | held_gnss | 38.604564 | — | 0 |
| mars_5 | 10 | adaptive_full | 0.934966 | 0.095197 | 3 |
| mars_5 | 10 | adaptive_gps_only | 2.286876 | 0.052945 | 3 |
| mars_5 | 10 | constant_velocity_gnss | 0.933140 | — | 0 |
| mars_5 | 10 | damped_cv | 0.266359 | — | 0 |
| mars_5 | 10 | held_gnss | 0.587165 | — | 0 |
| mars_5 | 30 | adaptive_full | 3.054385 | 0.241020 | 3 |
| mars_5 | 30 | adaptive_gps_only | 6.418177 | 0.133459 | 3 |
| mars_5 | 30 | constant_velocity_gnss | 3.485750 | — | 0 |
| mars_5 | 30 | damped_cv | 0.436568 | — | 0 |
| mars_5 | 30 | held_gnss | 0.599547 | — | 0 |
| mars_5 | 60 | adaptive_full | 10.507714 | 0.555168 | 3 |
| mars_5 | 60 | adaptive_gps_only | 14.770553 | 0.183980 | 3 |
| mars_5 | 60 | constant_velocity_gnss | 12.178496 | — | 0 |
| mars_5 | 60 | damped_cv | 6.760384 | — | 0 |
| mars_5 | 60 | held_gnss | 6.330691 | — | 0 |

## Gate behavior

![Every gate trace](outputs/insane_adaptive/gate_traces.png)

Each panel shows all six models on one fixed outage; traces include permitted history and recovery, shaded region is the outage. Gate observations describe prediction control, not a stop detector or causal explanation. Complete token timestamps, gate, residual velocity, integrated velocity/motion and native predictions are in NPZ.

| scenario | flight | duration_s | method | seed | g_mean | g_min | g_max | g_first | g_last |
|---|---|---|---|---|---|---|---|---|---|
| mars_4_e1_d10 | mars_4 | 10 | adaptive_full | 0 | 0.433528 | 0.427679 | 0.446241 | 0.446241 | 0.427930 |
| mars_4_e1_d10 | mars_4 | 10 | adaptive_gps_only | 0 | 0.309738 | 0.309177 | 0.310947 | 0.310934 | 0.309350 |
| mars_4_e1_d10 | mars_4 | 10 | adaptive_full | 1 | 0.387022 | 0.386140 | 0.389442 | 0.386845 | 0.386198 |
| mars_4_e1_d10 | mars_4 | 10 | adaptive_gps_only | 1 | 0.303972 | 0.303395 | 0.306057 | 0.304376 | 0.303395 |
| mars_4_e1_d10 | mars_4 | 10 | adaptive_full | 2 | 0.477322 | 0.464591 | 0.486096 | 0.464591 | 0.474592 |
| mars_4_e1_d10 | mars_4 | 10 | adaptive_gps_only | 2 | 0.322528 | 0.321875 | 0.323577 | 0.321930 | 0.321982 |
| mars_4_e1_d30 | mars_4 | 30 | adaptive_full | 0 | 0.414708 | 0.306397 | 0.563104 | 0.446241 | 0.419725 |
| mars_4_e1_d30 | mars_4 | 30 | adaptive_gps_only | 0 | 0.309060 | 0.308068 | 0.310947 | 0.310934 | 0.308115 |
| mars_4_e1_d30 | mars_4 | 30 | adaptive_full | 1 | 0.386635 | 0.376727 | 0.400658 | 0.386845 | 0.397429 |
| mars_4_e1_d30 | mars_4 | 30 | adaptive_gps_only | 1 | 0.303038 | 0.301744 | 0.306057 | 0.304376 | 0.301744 |
| mars_4_e1_d30 | mars_4 | 30 | adaptive_full | 2 | 0.451577 | 0.405622 | 0.486096 | 0.464591 | 0.422546 |
| mars_4_e1_d30 | mars_4 | 30 | adaptive_gps_only | 2 | 0.321473 | 0.319909 | 0.323577 | 0.321930 | 0.319975 |
| mars_4_e1_d60 | mars_4 | 60 | adaptive_full | 0 | 0.412578 | 0.306397 | 0.563104 | 0.446241 | 0.486428 |
| mars_4_e1_d60 | mars_4 | 60 | adaptive_gps_only | 0 | 0.308168 | 0.306183 | 0.310947 | 0.310934 | 0.306492 |
| mars_4_e1_d60 | mars_4 | 60 | adaptive_full | 1 | 0.390685 | 0.372763 | 0.415654 | 0.386845 | 0.410558 |
| mars_4_e1_d60 | mars_4 | 60 | adaptive_gps_only | 1 | 0.301880 | 0.299734 | 0.306057 | 0.304376 | 0.300214 |
| mars_4_e1_d60 | mars_4 | 60 | adaptive_full | 2 | 0.427620 | 0.373296 | 0.486096 | 0.464591 | 0.475809 |
| mars_4_e1_d60 | mars_4 | 60 | adaptive_gps_only | 2 | 0.320029 | 0.316974 | 0.323577 | 0.321930 | 0.317126 |
| mars_4_e2_d10 | mars_4 | 10 | adaptive_full | 0 | 0.484029 | 0.432200 | 0.523479 | 0.510370 | 0.480688 |
| mars_4_e2_d10 | mars_4 | 10 | adaptive_gps_only | 0 | 0.311027 | 0.310583 | 0.313118 | 0.313118 | 0.310592 |
| mars_4_e2_d10 | mars_4 | 10 | adaptive_full | 1 | 0.406671 | 0.396105 | 0.412922 | 0.411507 | 0.406708 |
| mars_4_e2_d10 | mars_4 | 10 | adaptive_gps_only | 1 | 0.301936 | 0.301463 | 0.304273 | 0.302945 | 0.301463 |
| mars_4_e2_d10 | mars_4 | 10 | adaptive_full | 2 | 0.497244 | 0.470668 | 0.522513 | 0.520120 | 0.493462 |
| mars_4_e2_d10 | mars_4 | 10 | adaptive_gps_only | 2 | 0.326211 | 0.325529 | 0.327890 | 0.326525 | 0.325541 |
| mars_4_e2_d30 | mars_4 | 30 | adaptive_full | 0 | 0.480008 | 0.423806 | 0.543734 | 0.510370 | 0.534026 |
| mars_4_e2_d30 | mars_4 | 30 | adaptive_gps_only | 0 | 0.310294 | 0.309248 | 0.313118 | 0.313118 | 0.309277 |
| mars_4_e2_d30 | mars_4 | 30 | adaptive_full | 1 | 0.401973 | 0.377776 | 0.413355 | 0.411507 | 0.408372 |
| mars_4_e2_d30 | mars_4 | 30 | adaptive_gps_only | 1 | 0.301096 | 0.299916 | 0.304273 | 0.302945 | 0.299918 |
| mars_4_e2_d30 | mars_4 | 30 | adaptive_full | 2 | 0.499434 | 0.470668 | 0.522513 | 0.520120 | 0.516340 |
| mars_4_e2_d30 | mars_4 | 30 | adaptive_gps_only | 2 | 0.325018 | 0.323295 | 0.327890 | 0.326525 | 0.323337 |
| mars_4_e2_d60 | mars_4 | 60 | adaptive_full | 0 | 0.464801 | 0.351298 | 0.563270 | 0.510370 | 0.496710 |
| mars_4_e2_d60 | mars_4 | 60 | adaptive_gps_only | 0 | 0.309348 | 0.307349 | 0.313118 | 0.313118 | 0.307575 |
| mars_4_e2_d60 | mars_4 | 60 | adaptive_full | 1 | 0.398149 | 0.371470 | 0.414001 | 0.411507 | 0.393468 |
| mars_4_e2_d60 | mars_4 | 60 | adaptive_gps_only | 1 | 0.300018 | 0.297982 | 0.304273 | 0.302945 | 0.298020 |
| mars_4_e2_d60 | mars_4 | 60 | adaptive_full | 2 | 0.488422 | 0.437952 | 0.524304 | 0.520120 | 0.482775 |
| mars_4_e2_d60 | mars_4 | 60 | adaptive_gps_only | 2 | 0.323431 | 0.320267 | 0.327890 | 0.326525 | 0.320401 |
| mars_5_e1_d10 | mars_5 | 10 | adaptive_full | 0 | 0.370165 | 0.353185 | 0.495165 | 0.380668 | 0.353429 |
| mars_5_e1_d10 | mars_5 | 10 | adaptive_gps_only | 0 | 0.309528 | 0.309152 | 0.310937 | 0.310937 | 0.309152 |
| mars_5_e1_d10 | mars_5 | 10 | adaptive_full | 1 | 0.425113 | 0.409550 | 0.480995 | 0.480995 | 0.422656 |
| mars_5_e1_d10 | mars_5 | 10 | adaptive_gps_only | 1 | 0.303752 | 0.303294 | 0.304290 | 0.304169 | 0.303294 |
| mars_5_e1_d10 | mars_5 | 10 | adaptive_full | 2 | 0.461875 | 0.440039 | 0.505418 | 0.440039 | 0.451021 |
| mars_5_e1_d10 | mars_5 | 10 | adaptive_gps_only | 2 | 0.321923 | 0.321368 | 0.322872 | 0.321463 | 0.321368 |
| mars_5_e1_d30 | mars_5 | 30 | adaptive_full | 0 | 0.355713 | 0.343499 | 0.495165 | 0.380668 | 0.344287 |
| mars_5_e1_d30 | mars_5 | 30 | adaptive_gps_only | 0 | 0.308863 | 0.307494 | 0.310937 | 0.310937 | 0.307555 |
| mars_5_e1_d30 | mars_5 | 30 | adaptive_full | 1 | 0.421993 | 0.409550 | 0.480995 | 0.480995 | 0.419493 |
| mars_5_e1_d30 | mars_5 | 30 | adaptive_gps_only | 1 | 0.302888 | 0.301654 | 0.304290 | 0.304169 | 0.302283 |
| mars_5_e1_d30 | mars_5 | 30 | adaptive_full | 2 | 0.447418 | 0.428953 | 0.505418 | 0.440039 | 0.429475 |
| mars_5_e1_d30 | mars_5 | 30 | adaptive_gps_only | 2 | 0.320881 | 0.319189 | 0.322872 | 0.321463 | 0.319189 |
| mars_5_e1_d60 | mars_5 | 60 | adaptive_full | 0 | 0.367366 | 0.304705 | 0.576010 | 0.380668 | 0.347967 |
| mars_5_e1_d60 | mars_5 | 60 | adaptive_gps_only | 0 | 0.307983 | 0.306282 | 0.310937 | 0.310937 | 0.306314 |
| mars_5_e1_d60 | mars_5 | 60 | adaptive_full | 1 | 0.410805 | 0.375993 | 0.480995 | 0.480995 | 0.391384 |
| mars_5_e1_d60 | mars_5 | 60 | adaptive_gps_only | 1 | 0.301735 | 0.299598 | 0.304290 | 0.304169 | 0.299600 |
| mars_5_e1_d60 | mars_5 | 60 | adaptive_full | 2 | 0.443973 | 0.397660 | 0.505418 | 0.440039 | 0.414662 |
| mars_5_e1_d60 | mars_5 | 60 | adaptive_gps_only | 2 | 0.319450 | 0.316670 | 0.322872 | 0.321463 | 0.316710 |

## Damped-CV selection

Tau = **5 s**, selected exclusively on 540 frozen train episodes before any adaptive validation metric. Fixed candidates: [1, 2, 5, 10, 20, 30, 60, 120]. Criterion: train macro relative-motion RMSE3D, smaller tau on ties. v(age)=v_CV exp(-age/tau), displacement=v_CV tau(1−exp(-age/tau)); age starts at last legal fix. This is a bounded baseline calibration, not a neural hyperparameter search.

| tau_s | train_macro_relative_rmse3d_m | episodes |
|---|---|---|
| 1.000000 | 20.903600 | 540 |
| 2.000000 | 20.608713 | 540 |
| 5.000000 | 20.342483 | 540 |
| 10.000000 | 20.863565 | 540 |
| 20.000000 | 22.444634 | 540 |
| 30.000000 | 23.698356 | 540 |
| 60.000000 | 25.794717 | 540 |
| 120.000000 | 27.358692 | 540 |

## Training and checks

The encoder and all original preprocessing/scenario/scaler hashes are preserved. Both variants have 7548 parameters. For each seed, initial tensor hashes and frozen training episode order match. AdamW lr0.001, weight_decay0.0001, batch4, clip1, Huber delta1m, max30 epochs, patience5, strict decrease in fixed validation macro RMSE, selected best checkpoint. Histories include all attempted epochs; same episode prefix is consumed until each model independently early-stops.

| model_seed | last_epoch | best_epoch | best_validation_criterion_m |
|---|---|---|---|
| adaptive_full_seed0 | 30 | 26 | 11.575646 |
| adaptive_gps_only_seed0 | 23 | 18 | 14.576295 |
| adaptive_full_seed1 | 23 | 18 | 12.542395 |
| adaptive_gps_only_seed1 | 23 | 18 | 14.576983 |
| adaptive_full_seed2 | 30 | 30 | 13.074474 |
| adaptive_gps_only_seed2 | 23 | 18 | 14.616175 |

Preflight validates source/calibration hashes, ENU projection, train-only scaler, all 540 train and 11 validation scenarios, exact native timestamps and masks. Mandatory original and adaptive tests cover gate0/gate1, irregular dt, state carry, future input perturbation, hidden GNSS mutation, shared inputs and backward gradients. Logs are in launch artifacts.

## Previous pilot — separate seed0 only

These saved results are descriptive context; they are never pooled with new seeds. No ungated seeds1/2 were trained, so a controlled three-seed estimate of the isolated gate effect is unavailable.

| flight | scenario | method | seed | relative_motion_rmse3d_m |
|---|---|---|---|---|
| mars_4 | mars_4_e1_d10 | full | 0.000000 | 2.023852 |
| mars_4 | mars_4_e1_d10 | gps_only | 0.000000 | 1.659391 |
| mars_4 | mars_4_e1_d30 | full | 0.000000 | 10.038588 |
| mars_4 | mars_4_e1_d30 | gps_only | 0.000000 | 9.466744 |
| mars_4 | mars_4_e1_d60 | full | 0.000000 | 28.533322 |
| mars_4 | mars_4_e1_d60 | gps_only | 0.000000 | 29.467426 |
| mars_4 | mars_4_e2_d10 | full | 0.000000 | 13.399618 |
| mars_4 | mars_4_e2_d10 | gps_only | 0.000000 | 14.145136 |
| mars_4 | mars_4_e2_d30 | full | 0.000000 | 30.238074 |
| mars_4 | mars_4_e2_d30 | gps_only | 0.000000 | 31.805966 |
| mars_4 | mars_4_e2_d60 | full | 0.000000 | 65.644896 |
| mars_4 | mars_4_e2_d60 | gps_only | 0.000000 | 67.847799 |
| mars_5 | mars_5_e1_d10 | full | 0.000000 | 1.704438 |
| mars_5 | mars_5_e1_d10 | gps_only | 0.000000 | 1.350168 |
| mars_5 | mars_5_e1_d30 | full | 0.000000 | 5.666687 |
| mars_5 | mars_5_e1_d30 | gps_only | 0.000000 | 4.628628 |
| mars_5 | mars_5_e1_d60 | full | 0.000000 | 16.218585 |
| mars_5 | mars_5_e1_d60 | gps_only | 0.000000 | 14.559854 |

## Metric / data / reference limitations

Native GT timestamps are scored without trajectory fitting. Primary is sqrt(mean(norm((pred−last legal GNSS)−(GT(t)−GT(last legal fix time)))²)) inside outage. Only the anchor is interpolated as a target/reference; no interpolated GT samples are counted as observations. Raw absolute3D/H/V, final unavailable error and recovery-first5s are saved. No GT attitude/position or RTK enters the encoder or initialization. Original timestamp corrections are already applied. GT is authors’ dual-RTK/magnetometer-derived PX4 IMU reference; ordinary GNSS antenna lever arm is missing. Filename 8hz does not guarantee 8 independent points/s. Test mars_6/7 remains absent from the Kaggle data.

ESKF remains **not_ready**, with no invented scores: missing ordinary-GNSS lever arm and validated absolute heading initialization. No superiority over INS/EKF is claimed. Validation selects checkpoints and is reused for this follow-up hypothesis after viewing the prior pilot; this is exploratory validation, not untouched confirmatory testing.

Environment: `{"python": "3.12.13", "torch": "2.10.0+cu128", "numpy": "2.0.2", "pandas": "2.3.3", "device": "cuda", "cuda": "12.8", "hardware": "Tesla T4"}`. Saved inference_ms is synchronized single forward/readout, not repeated benchmark; Held/CV share their joint timing. No training or full evaluation on Mac.

**STOP. No test, additional seeds/jobs, parameter changes or automatic next version.**

## Verified run identity

Kaggle version 1; output run ID 353123708; six completed trainings. Local integrity verification: 99 prediction files, maximum saved-metric difference 7.11e-15 m. No model forward, training or test access during verification.
