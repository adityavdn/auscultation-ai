# Results: heart and lung sound models

> Research prototype. Decision support only. Not a medical device.

Test sets are held out **by patient** where the dataset publishes patient IDs (CirCor, ICBHI, KAUH). CinC 2016 and Yaseen publish none, so they are split **by recording**; their scores may be optimistic.

Published references are given for context only: they use other test sets and are **not comparable** one-to-one with these numbers.

| Task | Data | Headline metric | Score | Macro-F1 | Test recordings (patients) | Published reference |
|---|---|---|---|---|---|---|
| heart_abnormal | PhysioNet/CinC 2016, normal vs abnormal (adult) | MAcc | 0.882 | 0.810 | 648 (648) | Challenge entries ~0.86 MAcc on a hidden test set (different data) |
| heart_murmur | CirCor DigiScope 2022, murmur present vs absent (children) | MAcc | 0.724 | 0.689 | 602 (175) | Challenge murmur task: top weighted accuracy ~0.78 (different metric) |
| heart_valve | Yaseen 2018, Normal/AS/MS/MR/MVP | Accuracy | 1.000 | 1.000 | 200 (200) | Paper ~97%. Small, clean, split by recording: 1.000 likely reflects near-duplicate clips, not real-world accuracy |
| lung_sound | ICBHI 2017, per breath cycle: normal/crackle/wheeze/both (official 60/40 split) | ICBHI score | 0.507 | 0.370 | 2756 (49) | Published state of the art ~0.60-0.65 ICBHI score |
| lung_diagnosis | ICBHI 2017 + KAUH 2021, patient diagnosis per recording | Macro-F1 | 0.433 | 0.433 | 252 (49) | No single standard benchmark; ICBHI is ~86% COPD, so macro-F1 is the honest number |

## heart_abnormal: confusion matrix (rows = truth, columns = prediction)

| | abnormal | normal |
|---|---|---|
| **abnormal** | 124 | 9 |
| **normal** | 87 | 428 |

## heart_murmur: confusion matrix (rows = truth, columns = prediction)

| | absent | present |
|---|---|---|
| **absent** | 382 | 97 |
| **present** | 43 | 80 |

## heart_valve: confusion matrix (rows = truth, columns = prediction)

| | AS | MR | MS | MVP | normal |
|---|---|---|---|---|---|
| **AS** | 40 | 0 | 0 | 0 | 0 |
| **MR** | 0 | 33 | 0 | 0 | 0 |
| **MS** | 0 | 0 | 37 | 0 | 0 |
| **MVP** | 0 | 0 | 0 | 35 | 0 |
| **normal** | 0 | 0 | 0 | 0 | 40 |

## lung_sound: confusion matrix (rows = truth, columns = prediction)

| | both | crackle | normal | wheeze |
|---|---|---|---|---|
| **both** | 14 | 24 | 78 | 27 |
| **crackle** | 5 | 222 | 405 | 17 |
| **normal** | 9 | 240 | 1152 | 178 |
| **wheeze** | 11 | 37 | 237 | 99 |

Sensitivity 0.285, specificity 0.730.

## lung_diagnosis: confusion matrix (rows = truth, columns = prediction)

| | asthma | copd | healthy | heart_failure | infection | other |
|---|---|---|---|---|---|---|
| **asthma** | 10 | 6 | 1 | 2 | 0 | 0 |
| **copd** | 0 | 158 | 3 | 0 | 3 | 0 |
| **healthy** | 2 | 5 | 20 | 0 | 1 | 0 |
| **heart_failure** | 6 | 0 | 5 | 1 | 0 | 0 |
| **infection** | 3 | 6 | 3 | 0 | 6 | 0 |
| **other** | 1 | 7 | 3 | 0 | 0 | 0 |

## Robustness

Adult model (CinC 2016) tested on children (CirCor): MAcc 0.491. A drop here is expected; it shows why local validation is needed before clinical use.

## Caveats

- CinC 2016 and Yaseen publish no patient IDs, so each recording is treated as its own patient.
- Datasets differ in stethoscope, population and labelling; results won't transfer unchanged to new devices.
- Heart and lung models are trained separately (no public paired data). The combined report uses transparent rules, not a joint model.

## Live check (14 Nov 2026, https://auscultation-ai.vercel.app)

All 35 test-split samples served by the live app were run through the public API: **33/35 agree with the dataset label**
(heart_abnormal 4/4, heart_murmur 3/4, heart_valve 8/8, lung_diagnosis 18/19). Median response 1.3 s, first request after idle up to ~15 s.
