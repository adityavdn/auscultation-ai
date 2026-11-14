"""metrics.json -> reports/results.md (results table with published benchmarks and caveats)."""
import json
import sys
from pathlib import Path

BENCH = {
    "heart_abnormal": ("PhysioNet/CinC 2016, normal vs abnormal (adult)", "MAcc", "macc",
                       "Challenge entries ~0.86 MAcc on a hidden test set (different data)"),
    "heart_murmur": ("CirCor DigiScope 2022, murmur present vs absent (children)", "MAcc", "macc",
                     "Challenge murmur task: top weighted accuracy ~0.78 (different metric)"),
    "heart_valve": ("Yaseen 2018, Normal/AS/MS/MR/MVP", "Accuracy", "accuracy",
                    "Paper ~97%. Small, clean, split by recording: 1.000 likely reflects near-duplicate clips, not real-world accuracy"),
    "lung_sound": ("ICBHI 2017, per breath cycle: normal/crackle/wheeze/both (official 60/40 split)", "ICBHI score",
                   "icbhi", "Published state of the art ~0.60-0.65 ICBHI score"),
    "lung_diagnosis": ("ICBHI 2017 + KAUH 2021, patient diagnosis per recording", "Macro-F1", "macro_f1",
                       "No single standard benchmark; ICBHI is ~86% COPD, so macro-F1 is the honest number"),
}


def _metric(r, key):
    v = r.get(key)
    return v["score"] if isinstance(v, dict) else v


def render(m):
    out = ["# Results: heart and lung sound models", "",
           "> Research prototype. Decision support only. Not a medical device.", "",
           "Test sets are held out **by patient** where the dataset publishes patient IDs (CirCor, ICBHI, KAUH). "
           "CinC 2016 and Yaseen publish none, so they are split **by recording**; their scores may be optimistic.", "",
           "Published references are given for context only: they use other test sets and are **not comparable** "
           "one-to-one with these numbers.", "",
           "| Task | Data | Headline metric | Score | Macro-F1 | Test recordings (patients) | Published reference |",
           "|---|---|---|---|---|---|---|"]
    for task, (data, name, key, ref) in BENCH.items():
        r = m.get(task, {})
        if "error" in r:
            out.append(f"| {task} | {data} | {name} | failed: {r['error']} | | | {ref} |")
            continue
        if not r:
            continue
        out.append(f"| {task} | {data} | {name} | {_metric(r, key):.3f} | {r['macro_f1']:.3f} | "
                   f"{r['n_test_rec']} ({r['n_test_patients']}) | {ref} |")
    for task in BENCH:
        r = m.get(task, {})
        if "confusion" not in r:
            continue
        out += ["", f"## {task}: confusion matrix (rows = truth, columns = prediction)", "",
                "| | " + " | ".join(r["labels"]) + " |", "|---" * (len(r["labels"]) + 1) + "|"]
        out += [f"| **{lab}** | " + " | ".join(map(str, row)) + " |" for lab, row in zip(r["labels"], r["confusion"])]
        if r.get("icbhi"):
            out.append(f"\nSensitivity {r['icbhi']['se']:.3f}, specificity {r['icbhi']['sp']:.3f}.")
    cross = m.get("cross_cinc_to_circor", {})
    if "macc" in cross:
        out += ["", "## Robustness", "", f"Adult model (CinC 2016) tested on children (CirCor): MAcc {cross['macc']:.3f}. "
                "A drop here is expected; it shows why local validation is needed before clinical use."]
    out += ["", "## Caveats", "",
            "- CinC 2016 and Yaseen publish no patient IDs, so each recording is treated as its own patient.",
            "- Datasets differ in stethoscope, population and labelling; results won't transfer unchanged to new devices.",
            "- Heart and lung models are trained separately (no public paired data). The combined report uses "
            "transparent rules, not a joint model."]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(render(json.loads(Path(sys.argv[1]).read_text())))
