from ml.make_report import render


def test_render_includes_tasks_and_errors():
    m = {"heart_abnormal": {"n": 10, "accuracy": 0.9, "macro_f1": 0.88, "macc": 0.87, "labels": ["abnormal", "normal"],
                            "confusion": [[4, 1], [0, 5]], "n_test_rec": 10, "n_test_patients": 10, "n_train_rec": 40},
         "lung_sound": {"error": "boom"}}
    md = render(m)
    assert "heart_abnormal" in md and "0.870" in md and "boom" in md and "Not a medical device" in md


def test_render_is_honest_about_record_level_splits_and_yaseen():
    md = render({"heart_valve": {"n": 1, "accuracy": 1.0, "macro_f1": 1.0, "labels": ["a"], "confusion": [[1]],
                                 "n_test_rec": 1, "n_test_patients": 1, "n_train_rec": 4}})
    assert "All test sets are held out **by patient**" not in md
    assert "by recording" in md and "not comparable" in md.lower()
