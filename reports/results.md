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

## Tried and not shipped: AudioSet-pretrained AST (14 to 15 Nov 2026)

The Audio Spectrogram Transformer (`MIT/ast-finetuned-audioset-10-10-0.4593`, frozen) turns each breath cycle into mean-pooled tokens from a few layers, and a light head trained on 20% held-out training patients picks the label. Each version was scored once on the official test split.

| Model | ICBHI score | Sensitivity | Specificity | CPU time per window (Apple M3) |
|---|---|---|---|---|
| LightGBM on hand-made features (live) | 0.508 | 0.288 | 0.729 | about 1 ms |
| Full AST, float32, 8 s input, layers 4/8/12 + logistic regression | 0.533 | 0.504 | 0.561 | 2,300 ms |
| Same head, AST quantised to int8 | 0.502 | 0.475 | 0.528 | 700 ms |
| Slim AST: 8 layers, 4 s input, per-channel int8, head trained on int8 embeddings | 0.519 | 0.497 | 0.540 | 70 ms |

The slim model is ten times faster than int8 and keeps most of the gain, but the gain is not proven. A paired bootstrap over the 49 test patients puts the slim model's advantage over LightGBM at +0.007, 95% interval -0.046 to +0.055; it wins in 63% of resamples. With this few test patients, ICBHI scores carry roughly ±0.05 of uncertainty, so all of these lung models are statistically level. What changes is the balance: AST finds far more abnormal cycles and raises more false alarms.

Nothing here went live. The slim model is published as an optional prerelease asset (`ast-slim-v1`) and the app can load it (`embedder` role in `app/inference.py`). A clear improvement probably needs AST fine-tuned end to end on a GPU, where published results reach 0.60 to 0.63.

## Caveats

- CinC 2016 and Yaseen publish no patient IDs, so each recording is treated as its own patient.
- Datasets differ in stethoscope, population and labelling; results won't transfer unchanged to new devices.
- Heart and lung models are trained separately (no public paired data). The combined report uses transparent rules, not a joint model.

## Live check (14 Nov 2026, https://auscultation-ai.vercel.app, after final review fixes)

All 34 held-out samples served by the live app were sent through the public API: **32/34 agree with the dataset label**
(heart_abnormal 4/4, heart_murmur 3/4, heart_valve 7/7, lung_diagnosis 18/19). Median response 1.6 s.
Non-stethoscope audio (pure tones, speech-like tones) is rejected with a clear message. Known gap: band-limited
white noise is not reliably rejected.
