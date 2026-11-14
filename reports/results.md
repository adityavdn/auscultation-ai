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

## Tried and not shipped: pretrained CNN for lung sounds (14 Nov 2026)

EfficientNet-B0 (ImageNet weights) fine-tuned on 128-band log-mel spectrograms of each breath cycle, 12 epochs on CPU, epoch picked on 15% of training patients. Scored on the same official test split (2,732 cycles, 49 patients; one clipped recording is now rejected, so 24 fewer cycles than the table above).

| Model | ICBHI score | Sensitivity | Specificity | Macro-F1 |
|---|---|---|---|---|
| LightGBM baseline (live) | 0.508 | 0.288 | 0.729 | 0.372 |
| EfficientNet-B0 CNN | 0.510 | 0.418 | 0.601 | 0.383 |
| 50/50 average of both (weight fixed before scoring) | 0.511 | 0.321 | 0.702 | 0.369 |

None of these is a meaningful gain, so the live app keeps the LightGBM model, which is smaller and faster. The CNN finds more abnormal cycles but raises more false alarms. Closing the gap to published results (~0.60-0.65) probably needs an audio-pretrained network (e.g. AudioSet) and a GPU.

## Tried and not shipped: AudioSet-pretrained AST embeddings (14 Nov 2026)

The Audio Spectrogram Transformer (`MIT/ast-finetuned-audioset-10-10-0.4593`, frozen) turned each breath cycle, repeat-padded to 8 s, into mean-pooled tokens from layers 4, 8 and 12 (2,304 values). A light head was chosen on 20% of training patients (logistic regression 0.594 vs LightGBM 0.585 on validation), refit on all training patients, and scored once on the official test split.

| Model | ICBHI score | Sensitivity | Specificity |
|---|---|---|---|
| LightGBM on hand-made features (live) | 0.508 | 0.288 | 0.729 |
| AST embeddings + logistic regression, float32 | **0.533** | 0.504 | 0.561 |
| Same head, AST quantised to int8 (dynamic) | 0.502 | 0.475 | 0.528 |

Float32 AST is the best lung model so far: it finds far more abnormal cycles at the cost of more false alarms. It is not live because it is slow on a CPU server: about 2.3 s per 3 s window in float32 and 0.7 s in int8 on an Apple M3, so one lung recording would take 8 to 28 s. Dynamic int8 loses the gain; per-channel int8 agrees with float32 on 97.5% of training cycles but is no faster. The code path is in place (`ml/run_lung_ast.py`, `ml/ast_fbank.py`, and the `embedder` role in `app/inference.py`), so it can be switched on when a GPU or a faster CPU is available.

## Caveats

- CinC 2016 and Yaseen publish no patient IDs, so each recording is treated as its own patient.
- Datasets differ in stethoscope, population and labelling; results won't transfer unchanged to new devices.
- Heart and lung models are trained separately (no public paired data). The combined report uses transparent rules, not a joint model.

## Live check (14 Nov 2026, https://auscultation-ai.vercel.app, after final review fixes)

All 34 held-out samples served by the live app were sent through the public API: **32/34 agree with the dataset label**
(heart_abnormal 4/4, heart_murmur 3/4, heart_valve 7/7, lung_diagnosis 18/19). Median response 1.6 s.
Non-stethoscope audio (pure tones, speech-like tones) is rejected with a clear message. Known gap: band-limited
white noise is not reliably rejected.
