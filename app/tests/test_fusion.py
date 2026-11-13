from app.fusion import fuse


def head(label, conf, low=None):
    return {"label": label, "confidence": conf, "probs": {}, "trained_on": "x",
            "low_confidence": conf < 0.6 if low is None else low}


HEART_OK = {"heart_abnormal": head("normal", 0.9), "heart_murmur": head("absent", 0.9),
            "heart_valve": head("normal", 0.9)}
LUNG_OK = {"lung_sound": head("normal", 0.9), "lung_diagnosis": head("healthy", 0.9)}


def rules(r):
    return {f["rule"] for f in r["flags"]}


def test_all_normal():
    r = fuse(HEART_OK, LUNG_OK)
    assert r["status"] == "normal" and not r["flags"]


def test_cardiac_pulmonary_flag():
    heart = HEART_OK | {"heart_murmur": head("present", 0.8)}
    lung = LUNG_OK | {"lung_sound": head("crackle", 0.8)}
    r = fuse(heart, lung)
    assert "cardiac_pulmonary" in rules(r) and r["status"] == "abnormal"


def test_airway_flag():
    r = fuse(HEART_OK, LUNG_OK | {"lung_sound": head("wheeze", 0.8)})
    assert "airway" in rules(r)


def test_low_conf_flag():
    r = fuse(HEART_OK | {"heart_abnormal": head("normal", 0.55)}, None)
    assert "quality" in rules(r) and r["status"] == "inconclusive"


def test_heart_only_ok():
    assert fuse(HEART_OK, None)["status"] == "normal"
