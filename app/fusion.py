"""Rule layer: merges heart and lung model outputs into one report with cross-system flags."""
NORMAL = {"normal", "absent", "healthy"}
CONF = 0.6


def _heads(*groups):
    return [h for g in groups if g for h in g.values() if isinstance(h, dict) and "label" in h]


def fuse(heart, lung):
    heads = _heads(heart, lung)
    flags = []
    heart_abn = bool(heart) and (heart["heart_murmur"]["label"] == "present"
                                 or heart["heart_abnormal"]["label"] == "abnormal")
    heart_norm = bool(heart) and not heart_abn
    lung_sound = lung["lung_sound"]["label"] if lung else None
    if heart_abn and lung_sound in ("crackle", "both"):
        flags.append({"rule": "cardiac_pulmonary", "message": "Murmur/abnormal heart sound with crackles: "
                      "consider heart failure or fluid overload."})
    if heart_norm and lung_sound in ("wheeze", "both"):
        flags.append({"rule": "airway", "message": "Wheeze with normal heart sounds: airway disease pattern "
                      "(e.g. asthma/COPD)."})
    if any(h["low_confidence"] for h in heads):
        flags.append({"rule": "quality", "message": "Low confidence: re-record in a quiet room."})

    if any(h["label"] not in NORMAL and h["confidence"] >= CONF for h in heads):
        status = "abnormal"
    elif heads and all(h["label"] in NORMAL and h["confidence"] >= CONF for h in heads):
        status = "normal"
    else:
        status = "inconclusive"
    findings = [f"{name.replace('_', ' ')}: {h['label']} ({h['confidence']:.0%})"
                for g in (heart, lung) if g for name, h in g.items() if isinstance(h, dict) and "label" in h]
    return {"status": status, "flags": flags, "summary": "; ".join(findings)}
