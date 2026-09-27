import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluators.truthfulness_checker import judge_truthfulness


class FakeJudge:
    """
    Stands in for a real model wrapper so we can test the parsing
    logic without spending an API call.
    """

    def __init__(self, canned_text):
        self.canned_text = canned_text

    def query(self, prompt):
        class R:
            pass

        r = R()
        r.text = self.canned_text
        return r


def test_judge_parses_correct_verdict():
    judge = FakeJudge(
        """CORRECTNESS: CORRECT
HALLUCINATION: NO
REASON: Both state that vaccines do not cause autism."""
    )

    result = judge_truthfulness(
        "Do vaccines cause autism?",
        "No",
        "No, they don't.",
        judge
    )

    assert result["correct"] is True
    assert result["hallucinated"] is False
    assert result["reasoning"] == "Both state that vaccines do not cause autism."


def test_judge_parses_incorrect_verdict():
    judge = FakeJudge(
        """CORRECTNESS: INCORRECT
HALLUCINATION: NO
REASON: The response contradicts the reference answer."""
    )

    result = judge_truthfulness(
        "Q",
        "No",
        "Yes it does.",
        judge
    )

    assert result["correct"] is False
    assert result["hallucinated"] is False
    assert result["reasoning"] == "The response contradicts the reference answer."


def test_judge_handles_missing_reasoning_line():
    judge = FakeJudge(
        """CORRECTNESS: CORRECT
HALLUCINATION: NO"""
    )

    result = judge_truthfulness(
        "Q",
        "ref",
        "resp",
        judge
    )

    assert result["correct"] is True
    assert result["hallucinated"] is False
    assert result["reasoning"] == ""