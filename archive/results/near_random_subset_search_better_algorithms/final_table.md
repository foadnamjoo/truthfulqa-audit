| target pairs | best method | mean actual pairs | mean held-out AUC | std | mean |AUC-0.5| | mean clean pairs | mean confounded fraction |
|---|---|---:|---:|---:|---:|---:|---:|
| 325 | hill_climb_from_score_rank | 326.0 | 0.5745 | 0.0527 | 0.0745 | 152.4 | 0.5325 |
| 350 | simulated_annealing_from_clean_first | 350.0 | 0.5694 | 0.0243 | 0.0694 | 235.7 | 0.3266 |
| 375 | beam_light_swaps_from_clean | 375.0 | 0.5586 | 0.0360 | 0.0631 | 236.6 | 0.3691 |
| 400 | simulated_annealing_from_clean_first | 400.0 | 0.5749 | 0.0269 | 0.0749 | 238.0 | 0.4050 |
| 425 | hill_climb_from_clean_first | 426.0 | 0.5988 | 0.0477 | 0.0995 | 238.8 | 0.4394 |

Optional:
| 300 | hill_climb_from_score_rank | 300.0 | 0.5644 | 0.0498 | 0.0685 | 145.1 | 0.5163 |
| 450 | hill_climb_from_clean_first | 450.0 | 0.6095 | 0.0481 | 0.1095 | 237.9 | 0.4713 |
