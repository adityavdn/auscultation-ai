# Results: heart and lung sound models

> Research prototype. Decision support only. Not a medical device.

All test sets are held out **by patient**: no patient appears in both training and testing.

| Task | Data | Headline metric | Score | Macro-F1 | Test recordings (patients) | Published reference |
|---|---|---|---|---|---|---|
| heart_abnormal | PhysioNet/CinC 2016, normal vs abnormal (adult) | MAcc | 0.882 | 0.810 | 648 (648) | Challenge winner ~0.86 MAcc (hidden test set) |
| heart_murmur | CirCor DigiScope 2022, murmur present vs absent (children) | MAcc | 0.724 | 0.689 | 602 (175) | Challenge murmur task: top weighted accuracy ~0.78 (different metric) |
| heart_valve | Yaseen 2018, Normal/AS/MS/MR/MVP | Accuracy | 1.000 | 1.000 | 200 (200) | Paper reports ~97%; dataset is small and clean, so scores flatter real use |
| lung_sound | ICBHI 2017, per breath cycle: normal/crackle/wheeze/both (official 60/40 split) | ICBHI score | 0.507 | 0.370 | 2756 (49) | Published state of the art ~0.60-0.65 ICBHI score |
| lung_diagnosis | ICBHI 2017 + KAUH 2021, patient diagnosis per recording | Macro-F1 | 0.215 | 0.215 | 252 (49) | No single standard benchmark; ICBHI is ~86% COPD, so macro-F1 is the honest number |

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

| | asthma | bronchiectasis | bronchiolitis | copd | healthy | heart_failure | lrti | other | pneumonia | urti |
|---|---|---|---|---|---|---|---|---|---|---|
| **asthma** | 9 | 0 | 0 | 1 | 6 | 1 | 0 | 1 | 0 | 0 |
| **bronchiectasis** | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| **bronchiolitis** | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| **copd** | 1 | 0 | 0 | 158 | 5 | 0 | 0 | 0 | 0 | 0 |
| **healthy** | 4 | 0 | 0 | 4 | 19 | 0 | 0 | 0 | 0 | 1 |
| **heart_failure** | 1 | 0 | 0 | 0 | 10 | 0 | 0 | 1 | 0 | 0 |
| **lrti** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| **other** | 3 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 |
| **pneumonia** | 2 | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| **urti** | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |

## Robustness

Adult model (CinC 2016) tested on children (CirCor): MAcc 0.491. A drop here is expected; it shows why local validation is needed before clinical use.

## Caveats

- CinC 2016 and Yaseen publish no patient IDs, so each recording is treated as its own patient.
- Datasets differ in stethoscope, population and labelling; results won't transfer unchanged to new devices.
- Heart and lung models are trained separately (no public paired data). The combined report uses transparent rules, not a joint model.
