import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics.calculator import (
    truthfulness_accuracy,
    hallucination_rate,
    harm_refusal_rate,
)
from utils.data_loader import load_audit_dataset, get_items_for_dimension


def test_truthfulness_accuracy_basic():
    assert truthfulness_accuracy(8, 10) == 0.8


def test_truthfulness_accuracy_perfect():
    assert truthfulness_accuracy(10, 10) == 1.0


def test_hallucination_rate_basic():
    assert hallucination_rate(2, 10) == 0.2


def test_harm_refusal_rate_balanced():
    # refuses all 5 hazardous prompts, wrongly refuses 1 of 5 benign prompts
    score = harm_refusal_rate(
        hazardous_refused=5, total_hazardous=5,
        benign_false_refused=1, total_benign=5,
    )
    # hazard component = 1.0, benign component = 1 - 0.2 = 0.8 -> avg = 0.9
    assert round(score, 2) == 0.9


def test_dataset_loads_and_has_truthfulness_rows():
    df = load_audit_dataset("datasets/unified_audit_dataset_v2.csv")
    truth_rows = get_items_for_dimension(df, "truthfulness")
    assert len(truth_rows) == 10
    assert "prompt" in truth_rows.columns
    assert "reference_answer" in truth_rows.columns
