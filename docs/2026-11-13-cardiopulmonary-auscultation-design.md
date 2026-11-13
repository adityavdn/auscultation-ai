# Design: AI Cardiopulmonary Auscultation Prototype (heart + lung sounds)

Date: 2026-11-13. Status: draft for approval.

## 1. Intent and why

Aditya is building a research prototype at a professor's request. If NHS reviewers like it, it's a route into NHS work. The prototype takes heart and lung sounds (stethoscope recordings), says whether each is normal or abnormal, names the most likely condition, and puts both together in one report. An AI agent lets a patient describe symptoms and get the results explained, then passes a summary to a doctor.

Who uses it: one doctor, taught by Aditya. They test it on a phone with recordings from public datasets, where the true answer is known. No real patients and no hardware yet. A digital/IoT stethoscope comes later and posts audio to the same API.

Success means:
1. **Today:** the professor sees trained models with honest, strong numbers on held-out data, plus a short results report.
2. **Tomorrow:** the doctor opens a link on their phone, picks or uploads a recording, and gets normal/abnormal, a likely condition and a confidence, and agrees with it most of the time.

## 2. Goals, non-goals, anti-goals

Goals
- Heart model: normal vs abnormal, plus a disease class where the data allows (AS, MS, MR, MVP).
- Lung model: normal / crackle / wheeze / both, plus patient-level diagnosis (COPD, pneumonia, URTI, bronchiectasis, bronchiolitis, asthma, heart failure, healthy, ...).
- A combined heart + lung report with cross-system flags.
- Mobile-first web app (installable PWA) on a public free host, with a documented REST API.
- An AI agent: patient symptom chat that leads to a doctor summary/inbox, plus plain-language explanation of results.
- Free throughout: data, compute, hosting, LLM.

Non-goals (for now)
- Real patient data, NHS login, GP Connect, EHR integration.
- Hardware integration. Only the API contract is defined for the later IoT device.
- A native iOS/Android app.
- MHRA/UKCA certification. The prototype is labelled research / decision support only.

Anti-goals (a failure even if it "works")
- Inflated accuracy from train/test leakage. Splits must be by patient, never by recording or cycle.
- The tool or the agent stating a diagnosis as fact, or giving treatment advice.
- A demo that breaks live: a slow cold start, or crashing on a phone browser.

## 3. Constraints

- Deadline: models and report today (13 Nov), app tomorrow (14 Nov).
- Everything free and open-licence. One free LLM API key, which Aditya creates (Groq; Gemini as fallback).
- Laptop: Apple M3, 8 GB RAM, 45 GB free disk. Aditya prefers cloud storage over the local disk.
- No paired heart + lung recordings exist publicly at useful scale. That shapes the fusion design (section 6).

## 4. Data (all free, no login unless noted)

| Use | Dataset | Size | Labels |
|---|---|---|---|
| Heart, normal/abnormal (adult) | PhysioNet/CinC Challenge 2016 | 3,240 recordings | normal / abnormal |
| Heart, murmur (paediatric) | CirCor DigiScope 2022 (PhysioNet) | 5,272 recordings, 1,568 patients | murmur present/absent/unknown, outcome |
| Heart, disease names | Yaseen et al. 2018 (GitHub) | 1,000 recordings | Normal, AS, MS, MR, MVP |
| Lung, sounds + diagnosis | ICBHI 2017 | 920 recordings, 126 patients | crackle/wheeze per cycle; diagnosis per patient |
| Lung, more diagnoses | KAUH / Fraiwan 2021 (Mendeley) | 336 recordings, 112 patients | asthma, heart failure, pneumonia, bronchitis, effusion, fibrosis, COPD, normal |
| Optional, both sounds | HLS-CMDS 2024 (manikin) | small | mixed heart+lung, used only for a demo of separation if time allows |

Known caveats, which go in the report:
- ICBHI is about 86% COPD.
- Yaseen is clean and small, so its near-perfect scores flatter it.
- CirCor is children.
- Datasets differ in device and sample rate. Everything gets resampled to 4 kHz (heart) and 4 kHz (lung, band 50–2000 Hz).

## 5. Models

Approach: try a fast baseline and a pretrained-audio model per task, and keep whichever scores best on **patient-wise** held-out data.

1. Baseline: log-mel + MFCC statistics, then gradient boosting (LightGBM). It's minutes to train and gives a strong floor.
2. Main: a pretrained audio network fine-tuned on log-mel spectrograms. Candidates are PANNs CNN14, AST/BEATs, or Google HeAR embeddings (free, gated) with a light head. Trained on the M3 (MPS) or a free cloud GPU.
3. Long recordings are split into fixed windows. Predictions are averaged per recording, then per patient.

Metrics:
- Sensitivity, specificity, their mean (the official ICBHI score), macro-F1, a confusion matrix, and a calibrated confidence.
- Reported against published benchmarks, honestly.

The report says what each number means for a doctor.

## 6. Combining heart and lung

There are no paired recordings, so we don't train one joint model.
- Each model runs on its own input.
- A transparent rule layer merges the outputs into one report and adds cross-system flags. Examples:
  - murmur + crackles: "consider heart failure / fluid overload"
  - normal heart + wheeze: "airway disease pattern"
- KAUH's heart-failure-from-lung-sounds label backs one of these flags.
- Each flag cites its rule, so a doctor can check it.

## 7. App and API (approach A, chosen by Aditya)

- Backend: Python FastAPI on Hugging Face Spaces (free, Docker). Endpoints:
  - `POST /api/analyze/heart`
  - `POST /api/analyze/lung`
  - `POST /api/analyze/combined`
  - `POST /api/chat`
  - `GET/POST /api/cases`
  - `GET /api/samples`
  - OpenAPI docs at `/docs`
- The later IoT stethoscope posts audio to the same analyze endpoints.
- Frontend: mobile-first PWA served by the same app. Screens:
  - Analyse (pick a dataset sample or upload or record audio, then see the result, a spectrogram with the parts that drove the decision highlighted, and confidence)
  - Patient chat
  - Doctor inbox
- Agent: an LLM via free API.
  - The patient chat collects symptoms, explains results in plain words, never diagnoses, and escalates red flags ("call 999 / 111").
  - It produces a structured summary that lands in the doctor's inbox.
- Storage: SQLite. On the free host it resets on restart, which is acceptable for a prototype.
- Every screen carries the banner "Research prototype. Decision support only. Not a medical device."

## 8. Approaches considered

- **A. FastAPI + PWA on HF Spaces (chosen):** looks like a product, has a real API, is free, and works on any phone. Cost: about 30 s cold start after idle. Mitigation: open it before the demo.
- B. A Gradio app: faster to build, but looks like a lab tool and handles multi-screen flows badly. Rejected.
- C. A native app with the model on the device: weeks of work and a paid iOS account. Rejected for the prototype.

## 9. Left to the builder

The exact network architecture and hyperparameters, the window length, the UI styling details, the rule thresholds, and which optional datasets get used if download or time limits bite.

## 10. Builder's calls (made by Claude; Aditya can override)

- **Scope today:** the LightGBM baseline on every task comes first and is the guaranteed deliverable. The pretrained net is a stretch goal. Whichever scores better ships.
- **Evaluation:** patient-wise held-out within each dataset is the headline. ICBHI uses its official 60/40 patient split. One extra robustness number: train on CinC2016, test on CirCor (adult to child, cross-device), reported honestly even if low.
- **Benchmarks:**
  - ICBHI official score, mean of Se and Sp. Published SOTA is about 60–65% (4-class).
  - CinC2016 MAcc. Top entries reach about 86%.
  - CirCor murmur weighted accuracy, as in the 2022 challenge.
  - Yaseen 5-class accuracy, flagged as a small and clean dataset.
  - No hard pass mark. The aim is to match or approach published numbers, without leakage.
- **Heart heads, kept separate because the label sets don't line up:**
  - (a) normal/abnormal, from CinC2016
  - (b) murmur present/absent, from CirCor
  - (c) valve disease (Normal/AS/MS/MR/MVP), from Yaseen
  - The app shows all three, each with the dataset it was trained on.
- **Lung heads:**
  - (a) per-cycle sound type, from ICBHI
  - (b) recording-level diagnosis, from ICBHI + KAUH pooled. A dataset-ID shortcut check is reported.
- **Serving budget:**
  - Total model size under 100 MB, CPU inference under 3 s per recording on the free host.
  - Small CNNs or LightGBM. Export to ONNX if needed.
  - Google HeAR is dropped, because gated approval could take days.
- **Open-licence mirrors on the Hugging Face Hub** are used where the original site is down. The ICBHI official site timed out today, and an HF mirror exists.

## 11. Decisions by Aditya (13 Nov)

- **Training runs in Kaggle notebooks** (free GPU), not on the laptop. Data stays in Kaggle/HF. Code goes in a private GitHub repo. Weights go on the HF Hub.
- Aditya has HF and Groq accounts, and pastes keys himself.
- Design approved 13 Nov 2026.
