# Cardiopulmonary Auscultation AI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train heart and lung sound classifiers on free public datasets, then serve them with a combined report and an AI patient/doctor agent through a FastAPI + mobile PWA on Hugging Face Spaces.

**Architecture:** One repo with two halves.
- `ml/` is pure-Python feature, data and training code. It runs inside Kaggle notebooks (pushed with the `kaggle` CLI) and writes `models/*.joblib|onnx` and `reports/metrics.json`.
- `app/` is a FastAPI service. It loads those models, applies a rule-based fusion layer, calls a Groq LLM for chat, and serves a static PWA. It deploys as a Docker Space.

**Tech Stack:** Python 3.11, numpy, scipy, librosa, soundfile, scikit-learn, lightgbm, joblib, PyTorch (stretch CNN), FastAPI, uvicorn, python-multipart, httpx, pytest, vanilla HTML/JS/CSS PWA, ffmpeg (in Docker), Kaggle CLI, huggingface_hub.

**Spec:** `docs/2026-11-13-cardiopulmonary-auscultation-design.md`

## Global Constraints

- Free only. No paid API, GPU or host.
- Splits are **patient-wise**. ICBHI uses its official 60/40 patient split. No patient may appear in both train and test.
- Resample all audio to **4000 Hz mono**. Band-pass **25–400 Hz for heart**, **50–1900 Hz for lung**. 1900 is just below Nyquist at 4 kHz.
- Total served model size **< 100 MB**. CPU inference **< 3 s** per recording.
- Every UI screen and every API analyze response carries: `"Research prototype. Decision support only. Not a medical device."`
- The agent never states a diagnosis as fact and never gives treatment or dosing. Red-flag symptoms (chest pain, severe breathlessness, blue lips, fainting, coughing blood) → respond with a 999/111 escalation.
- Keys (Kaggle, HF, Groq) come from env vars or files Aditya creates. Code never hardcodes them, and Claude never types them.
- Repo: private GitHub `adityavdn/auscultation-ai`. Commits are authored as adityavdn, with no Claude co-author line.
- The local laptop holds code only, in the scratchpad. Data and training live on Kaggle. Weights live on the HF Hub.

## Review Focus

1. **Phone recordings arrive as webm/opus, m4a or mp3, not wav.** The app should decode them (ffmpeg via librosa/audioread), not crash. Test in Task 7.
2. **Very short (<2 s), silent or clipped audio.** Expect a 422 with a clear message, not a NaN prediction. Test in Task 2 (`load_audio`) and Task 7.
3. **Stereo, 44.1/48 kHz, or huge (>20 MB) uploads.** Downmix and resample. Reject files over 20 MB with a 413. Test in Task 7.
4. **No Groq key, or Groq rate-limited.** Analysis still works, and chat returns a polite "assistant unavailable" message. Test in Task 8.
5. **Low-confidence or out-of-distribution input (a heart sound sent to the lung endpoint).** Expect a "low confidence, re-record" warning when the top probability is < 0.6. Test in Task 6.

---

### Task 1: Repo skeleton + metrics

**Files:**
- Create: `ml/metrics.py`, `ml/tests/test_metrics.py`, `requirements-ml.txt`, `.gitignore` (ignore `data/`, `*.wav`, `models/`), `README.md`

**Interfaces:**
- Produces: `sensitivity_specificity(y_true, y_pred, positive_labels) -> tuple[float, float]`, `icbhi_score(y_true, y_pred, normal_label="normal") -> dict` (keys `se`, `sp`, `score`), `macc(y_true, y_pred) -> float` (mean of Se and Sp for binary), `report(y_true, y_pred, labels) -> dict` (adds `macro_f1`, `confusion` as a nested list).

- [ ] **Step 1:** `gh repo create adityavdn/auscultation-ai --private`, then clone into the scratchpad.
- [ ] **Step 2: Failing test**

```python
def test_icbhi_score():
    yt = ["normal","crackle","wheeze","both","normal"]
    yp = ["normal","crackle","normal","both","crackle"]
    r = icbhi_score(yt, yp)
    assert r["sp"] == 0.5            # 1 of 2 normals right
    assert abs(r["se"] - 2/3) < 1e-9 # 2 of 3 abnormal exactly right
    assert abs(r["score"] - (0.5 + 2/3)/2) < 1e-9
def test_macc_binary():
    assert macc([0,0,1,1],[0,1,1,1]) == 0.75
```

- [ ] **Step 3:** `pytest ml/tests/test_metrics.py`. Expect FAIL (import error).
- [ ] **Step 4:** Implement. ICBHI Se = exact-class matches on abnormal ÷ abnormal count. Sp = normal correct ÷ normal count.
- [ ] **Step 5:** pytest passes. Commit `feat: metrics`.

### Task 2: Audio loading + features

**Files:** Create `ml/audio.py`, `ml/features.py`, `ml/tests/test_audio.py`

**Interfaces:**
- Produces:
  - `load_audio(path_or_bytes, kind: Literal["heart","lung"]) -> np.ndarray` (float32, 4000 Hz, band-passed, peak-normalised). Raises `AudioError` if duration < 2.0 s, or if RMS < 1e-4 (silent).
  - `windows(x, sr=4000, win_s, hop_s) -> list[np.ndarray]`: heart 5 s / hop 2.5 s, lung 8 s / hop 4 s. The last partial window is zero-padded if it's ≥ 50% full.
  - `logmel(x) -> np.ndarray[64, T]`: n_fft 256, hop 64, 64 mels, fmax 2000.
  - `stat_features(x, kind) -> np.ndarray[D]`: mean/std/skew of 40 MFCC + deltas, spectral centroid/bandwidth/rolloff/flatness, ZCR, and per-band energy ratios. Heart adds an S1/S2 envelope periodicity feature: the autocorrelation peak of the Hilbert envelope in the 0.4–1.5 s lag range.

- [ ] **Step 1: Failing tests**
  - `test_resample_and_mono`: stereo 44.1 kHz 3 s sine → length 12000, ndim 1.
  - `test_short_raises`: 1.5 s → `AudioError`.
  - `test_silent_raises`: zeros → `AudioError`.
  - `test_windows_count`: 12 s heart → 4 windows.
  - `test_stat_features_finite`: `np.isfinite(stat_features(noise,"heart")).all()`, with a fixed length D.
- [ ] **Step 2:** Run. Expect FAIL.
- [ ] **Step 3:** Implement with librosa (`librosa.load(sr=4000, mono=True)` handles any ffmpeg format) and scipy `butter(4, band, "bandpass", fs=4000)` + `sosfiltfilt`.
- [ ] **Step 4:** pytest passes. Commit `feat: audio features`.

### Task 3: Dataset loaders + patient-wise splits

**Files:** Create `ml/datasets.py`, `ml/tests/test_datasets.py` (fixtures: tiny fake directory trees built in `tmp_path`)

**Interfaces:**
- Produces: each loader returns a `pd.DataFrame` with columns `path, patient_id, label, dataset, split` (`split` ∈ {train, test}). Test is 20% unless an official split exists. Seed 42, `GroupShuffleSplit` on `patient_id`, stratified where possible via `StratifiedGroupKFold(5)`, taking fold 0 as test.
  - `load_cinc2016(root)`: `training-*/REFERENCE.csv`, −1 → `normal`, 1 → `abnormal`. patient_id = record name (no patient IDs are published; noted in the report).
  - `load_circor(root)`: `training_data.csv` + `training_data/{pid}_{loc}.wav`. Label = Murmur Present/Absent. Drop Unknown.
  - `load_yaseen(root)`: folders `N, AS, MS, MR, MVP` → labels `normal, AS, MS, MR, MVP`. patient_id = file stem (flagged).
  - `load_icbhi_cycles(root)`: one row per respiratory cycle with `start, end` columns. Label from the crackle/wheeze bits → `normal|crackle|wheeze|both`. Split from the official `ICBHI_challenge_train_test.txt`. If the mirror lacks it, use a patient-wise 60/40 and say so in the report.
  - `load_lung_diagnosis(icbhi_root, kauh_root|None)`: one row per recording, label = the patient diagnosis lower-cased and mapped onto `{healthy, copd, pneumonia, urti, bronchiectasis, bronchiolitis, lrti, asthma, heart_failure, other}`. patient_id is prefixed with the dataset name.

- [ ] **Step 1: Failing test** `test_no_patient_leak` for every loader on the fixture trees: `set(train.patient_id).isdisjoint(test.patient_id)`. Plus `test_icbhi_cycle_labels`: a `.txt` line `0.1 1.2 1 1` gives `both`.
- [ ] **Step 2:** Run. Expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** pytest passes. Commit `feat: dataset loaders`.

### Task 4: Baseline training on Kaggle → today's deliverable

**Files:** Create `ml/train_baseline.py`, `kaggle/kernel-metadata.json`, `kaggle/run.py` (downloads the datasets, then runs `train_baseline`), `ml/tests/test_train_smoke.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: `train_task(df, kind, task_name) -> dict` (fits LightGBM on window-level `stat_features`, averages window probabilities per recording, writes `models/{task_name}.joblib` containing `{"model", "labels", "kind"}`, returns the `report()` dict). Writes `reports/metrics.json` keyed by the task names `heart_abnormal, heart_murmur, heart_valve, lung_sound, lung_diagnosis`.
- Download sources:
  - PhysioNet: `wget -r -np` of the `challenge-2016/1.0.0/training-*` folders, and `circor-heart-sound/1.0.3`.
  - Yaseen: git clone.
  - ICBHI: the HF mirror (`huggingface_hub.snapshot_download`). Pick the mirror that has raw wav + txt + the diagnosis file. Check `okchang95/ICBHI2017-raw` first.
  - KAUH: Mendeley `data.mendeley.com/datasets/jwyy9np4gv` zip. Skip if the download fails.

- [ ] **Step 1: Failing smoke test.** 20 synthetic clips (two sine frequencies = two classes, two patients per class in test) → `train_task` returns `macro_f1 > 0.9` and writes the joblib.
- [ ] **Step 2:** Implement. Use `class_weight="balanced"` and `n_estimators=400`, `learning_rate=0.05`, `num_leaves=31`. Run a 5-fold patient-grouped CV on train to pick the decision threshold for the binary tasks (max MAcc).
- [ ] **Step 3:** Aditya puts `kaggle.json` in `~/.kaggle/` himself. Then run `kaggle kernels push -p kaggle/` with `enable_internet: true` and GPU off (LightGBM is CPU). Poll `kaggle kernels status` until `complete`, then fetch outputs with `kaggle kernels output -p scratchpad/out`.
- [ ] **Step 4:** Check `metrics.json` has all five tasks, each with no leakage. Upload `models/` to the private HF model repo `adityavdn/auscultation-models` (needs the HF token Aditya provides).
- [ ] **Step 5:** Write `reports/results.md` (a table vs the published benchmarks from spec §10, with caveats) and the confusion-matrix PNGs. Commit `feat: baseline models + results`.

### Task 5 (stretch): Small CNN on log-mel

**Files:** Create `ml/train_cnn.py`. Add a GPU-enabled mode to `kaggle/run.py`.

**Interfaces:** `train_cnn(df, kind, task_name) -> dict`. Same outputs as Task 4, saved as `models/{task_name}.onnx` with `labels` in a sidecar `.json`. The architecture is 4 conv blocks (32-64-128-128) with BN, ReLU and max-pool, then global average pooling and a linear layer: under 1 MB. SpecAugment, mixup 0.2, 40 epochs, AdamW 1e-3, cosine schedule.

- [ ] **Step 1:** Smoke test on the synthetic set (macro_f1 > 0.9), CPU, 3 epochs.
- [ ] **Step 2:** Run on Kaggle with GPU T4.
- [ ] **Step 3:** Per task, keep whichever of the two models has the higher headline metric. Record the choice in `metrics.json["chosen"]`. Commit.

### Task 6: Inference + fusion

**Files:** Create `app/inference.py`, `app/fusion.py`, `app/tests/test_fusion.py`

**Interfaces:**
- Consumes: `ml/audio.py`, `ml/features.py` (copied into the image), `models/`.
- Produces:
  - `analyze(audio_bytes, kind) -> dict`. Heart returns `{"heart_abnormal": {...}, "heart_murmur": {...}, "heart_valve": {...}}`. Lung returns `{"lung_sound": {...}, "lung_diagnosis": {...}}`. Each `{...}` is `{"label", "confidence", "probs": {label: p}, "trained_on": str, "low_confidence": bool}`. Results also include `"spectrogram_png_b64"` and `"window_scores"` (a per-window abnormal probability, for highlighting).
  - `fuse(heart: dict|None, lung: dict|None) -> {"summary": str, "status": "normal"|"abnormal"|"inconclusive", "flags": [{"rule", "message"}]}`.
- Rules (exact):
  - murmur Present **or** heart abnormal, **and** lung crackle/both → `"cardiac_pulmonary"`: "Murmur/abnormal heart sound with crackles: consider heart failure or fluid overload."
  - heart normal + lung wheeze/both → `"airway"`: "Wheeze with normal heart sounds: airway disease pattern (e.g. asthma/COPD)."
  - any `low_confidence` → `"quality"`: "Low confidence: re-record in a quiet room."
  - Status is `abnormal` if any head is non-normal with confidence ≥ 0.6, `normal` if all are normal with confidence ≥ 0.6, else `inconclusive`.

- [ ] **Step 1: Failing tests:** one per rule, plus `test_low_conf_flag` (a top prob of 0.55 sets `low_confidence` and the status is `inconclusive`).
- [ ] **Step 2:** Implement.
- [ ] **Step 3:** pytest passes. Commit.

### Task 7: FastAPI service

**Files:** Create `app/main.py`, `app/db.py` (sqlite3 stdlib, tables `cases`, `messages`), `app/tests/test_api.py`, `Dockerfile` (python:3.11-slim + ffmpeg, port 7860), `requirements.txt`

**Interfaces:**
- Endpoints:
  - `POST /api/analyze/{kind}` (multipart `file`) → the `analyze()` dict + `disclaimer`.
  - `POST /api/analyze/combined` (multipart `heart`, `lung`, both optional, at least one required) → both + `fuse()`.
  - `GET /api/samples` → a list of `{id, kind, true_label, url}`: 3 test-split recordings per class, shipped in `app/samples/`, never from the train split.
  - `GET /api/samples/{id}/audio`.
  - `POST /api/cases` and `GET /api/cases`, `GET /api/cases/{id}` (doctor inbox).
  - `/docs` is automatic.
- Errors: an `AudioError` → 422 `{"detail": msg}`. Upload > 20 MB → 413. Unknown kind → 404.

- [ ] **Step 1: Failing tests** (TestClient, models stubbed by a fixture):
  - `test_analyze_wav_ok`
  - `test_analyze_webm_ok` (fixture generated with ffmpeg)
  - `test_short_audio_422`
  - `test_too_big_413`
  - `test_combined_requires_one`
  - `test_disclaimer_present`
- [ ] **Step 2:** Implement.
- [ ] **Step 3:** pytest passes. `docker build` succeeds. Commit.

### Task 8: AI agent

**Files:** Create `app/agent.py`, `app/tests/test_agent.py`

**Interfaces:**
- Produces:
  - `chat(history: list[dict], analysis: dict|None, role: "patient"|"doctor") -> {"reply": str, "red_flag": bool}`
  - `summarize_case(history, analysis) -> dict` (`{"symptoms", "duration", "history", "ai_findings", "urgency": "routine"|"soon"|"urgent"}`, via JSON mode).
  - Endpoints `POST /api/chat` and `POST /api/cases/from-chat` are added to `main.py`.
- LLM: Groq OpenAI-compatible `https://api.groq.com/openai/v1/chat/completions`, model `llama-3.3-70b-versatile`, called with httpx, key from `GROQ_API_KEY`. The system prompt bakes in the Global Constraints agent rules.
- A red-flag keyword pre-check runs **before** the LLM call. If it hits, return a fixed 999/111 message without calling the LLM.

- [ ] **Step 1: Failing tests:**
  - `test_red_flag_short_circuits` ("crushing chest pain" → `red_flag` is True and the LLM mock is not called).
  - `test_no_key_graceful` (`GROQ_API_KEY` unset → reply contains "unavailable", no exception).
  - `test_summary_schema` (mocked LLM JSON → has all keys).
- [ ] **Step 2:** Implement.
- [ ] **Step 3:** pytest passes. Commit.

### Task 9: Mobile PWA

**Files:** Create `app/static/index.html`, `app.js`, `styles.css`, `manifest.json`, `sw.js`, `icon.svg`

Three tabs:
- **Analyse:** heart/lung toggle; sample picker, file upload, or `MediaRecorder` record; result card with a status pill, each head's label and confidence bar, "trained on" text, the spectrogram with highlighted high-abnormal windows, fusion flags, the disclaimer, and a "Send to doctor" button.
- **Patient chat:** chat that carries the latest analysis, and a "Send summary to doctor" button.
- **Doctor inbox:** a list of cases with urgency badges, opening into the detail view.

The page is mobile-first, about 380 px wide, and works offline only as a shell.

- [ ] **Step 1:** Serve it via `StaticFiles` at `/`.
- [ ] **Step 2:** Verify in the browser pane at mobile size (375×812):
  - the sample → result flow
  - recording permission
  - the chat red-flag flow
  - an inbox case appears
  - no console errors
- [ ] **Step 3:** Commit.

### Task 10: Deploy

- [ ] **Step 1:** Create the HF Docker Space `adityavdn/auscultation-ai` (public, so the doctor's phone needs no login). Add secrets `GROQ_API_KEY` and `HF_TOKEN` via the Space settings. Aditya pastes these himself.
- [ ] **Step 2:** At Docker build time, `snapshot_download` models from `adityavdn/auscultation-models`.
- [ ] **Step 3:** Push. Open the Space URL in the browser pane at mobile size and run the Task 9 checklist on the live URL. Time a cold start and one analyze call (target < 3 s).
- [ ] **Step 4:** Add a short doctor quick-start to `README.md`. Commit and push.
