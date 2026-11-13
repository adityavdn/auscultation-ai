from ml.make_report import render


def test_render_includes_tasks_and_errors():
    m = {"heart_abnormal": {"n": 10, "accuracy": 0.9, "macro_f1": 0.88, "macc": 0.87, "labels": ["abnormal", "normal"],
                            "confusion": [[4, 1], [0, 5]], "n_test_rec": 10, "n_test_patients": 10, "n_train_rec": 40},
         "lung_sound": {"error": "boom"}}
    md = render(m)
    assert "heart_abnormal" in md and "0.870" in md and "boom" in md and "Not a medical device" in md
