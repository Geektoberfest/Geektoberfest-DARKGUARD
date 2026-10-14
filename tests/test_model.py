from model import (
    clean_text,
    evaluate_model,
    load_dataset,
    predict_text,
    train_model,
)


def test_clean_text():
    assert clean_text("  HELLO    WORLD  ") == "hello world"


def test_dataset_has_expected_columns():
    df=load_dataset()
    assert "text" in df.columns
    assert "label" in df.columns
    assert len(df)>0


def test_model_trains():
    results=train_model()
    assert 0<=results["accuracy"]<=1
    assert results["feature_count"]>0


def test_prediction_returns_valid_class():
    results=train_model()
    prediction,confidence,probabilities=predict_text(
        "Hurry! Only 2 seats left!",
        results["model"],
        results["vectorizer"]
    )
    assert prediction in (0,1)
    assert 0<=confidence<=1
    assert len(probabilities)==2


def test_evaluate_model():
    eval_res = evaluate_model(test_size=0.20, random_state=42)
    assert 0.90 <= eval_res["accuracy"] <= 1.0
    assert 0.85 <= eval_res["precision"] <= 1.0
    assert 0.85 <= eval_res["recall"] <= 1.0
    assert 0.85 <= eval_res["f1"] <= 1.0
    assert eval_res["confusion_matrix"].shape == (2, 2)
    assert "category_metrics" in eval_res
    assert len(eval_res["category_metrics"]) > 0
    assert len(eval_res["false_positives"]) > 0
    assert len(eval_res["false_negatives"]) > 0
    assert len(eval_res["recommended_weaknesses"]) == 2


def test_evaluate_model_reproducibility():
    run1 = evaluate_model(test_size=0.20, random_state=42)
    run2 = evaluate_model(test_size=0.20, random_state=42)
    assert run1["accuracy"] == run2["accuracy"]
    assert (run1["confusion_matrix"] == run2["confusion_matrix"]).all()
    assert len(run1["false_positives"]) == len(run2["false_positives"])
    assert len(run1["false_negatives"]) == len(run2["false_negatives"])


def test_evaluate_model_custom_split():
    eval_res = evaluate_model(test_size=0.15, random_state=123)
    assert 0.85 <= eval_res["accuracy"] <= 1.0
    assert eval_res["test_size"] == int(len(eval_res["test_df"]))


def test_evaluate_model_with_prefit_model():
    trained = train_model()
    eval_res = evaluate_model(
        model=trained["model"],
        vectorizer=trained["vectorizer"],
        test_size=0.20,
        random_state=42,
    )
    assert 0.90 <= eval_res["accuracy"] <= 1.0
    assert eval_res["confusion_matrix"].shape == (2, 2)


def test_evaluate_cli_run(tmp_path):
    from evaluate import run_evaluation
    out_file = tmp_path / "test_report.md"
    results = run_evaluation(save_report=True, output_path=str(out_file))
    assert 0.90 <= results["accuracy"] <= 1.0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "DARKGUARD Error Analysis" in content
    assert "Confusion Matrix" in content


