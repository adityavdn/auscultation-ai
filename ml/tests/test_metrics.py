from ml.metrics import icbhi_score, macc, report


def test_icbhi_score():
    yt = ["normal", "crackle", "wheeze", "both", "normal"]
    yp = ["normal", "crackle", "normal", "both", "crackle"]
    r = icbhi_score(yt, yp)
    assert r["sp"] == 0.5
    assert abs(r["se"] - 2 / 3) < 1e-9
    assert abs(r["score"] - (0.5 + 2 / 3) / 2) < 1e-9


def test_macc_binary():
    assert macc([0, 0, 1, 1], [0, 1, 1, 1]) == 0.75


def test_report_has_f1_and_confusion():
    r = report(["a", "b", "a"], ["a", "b", "b"], labels=["a", "b"])
    assert r["confusion"] == [[1, 1], [0, 1]]
    assert 0 < r["macro_f1"] < 1
