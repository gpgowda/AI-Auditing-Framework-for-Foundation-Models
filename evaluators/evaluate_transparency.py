"""
===============================================================================
FOUNDATION MODEL AUDITING FRAMEWORK
TRANSPARENCY EVALUATION MODULE
===============================================================================

Stage 2 for the Transparency dimension.

This module reads:

    outputs/{model}_transparency_responses.csv

produced by main.py and evaluates each model response using an LLM-as-a-Judge.

Each response is evaluated on TWO independent criteria:

    1. Correctness
       Does the model's answer match the reference answer?

    2. Expected Transparency Behavior
       Did the model demonstrate the specific transparency behavior
       required for that dataset sample?

The two scores are then combined to calculate:

    EQS : Explanation Quality Score

Output:

    outputs/{model}_transparency_judged.csv

Supported models:

    - Gemini
    - GPT
    - Llama

Claude is currently excluded according to config/settings.yaml.
===============================================================================
"""

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from evaluators.llm_judge import judge_explanation_quality

from metrics.calculator import (
    explanation_correctness_score,
    explanation_behavior_score,
    explanation_quality_score,
)

from models.gpt_wrapper import GPTWrapper


# ==============================================================================
# PROJECT CONFIGURATION
# ==============================================================================

MODEL_NAMES = [
    "gemini",
    "gpt",
    "llama",
]

OUTPUTS_DIR = Path("outputs")


# ==============================================================================
# HELPER FUNCTION
# ==============================================================================

def is_valid_response(response):
    """
    Checks whether a model response is present and usable.

    Returns:
        True  -> valid response exists
        False -> response is missing, empty, or NaN
    """

    if response is None:
        return False

    if pd.isna(response):
        return False

    if not isinstance(response, str):
        response = str(response)

    return bool(response.strip())


# ==============================================================================
# TRANSPARENCY EVALUATION
# ==============================================================================

def evaluate_transparency():
    """
    Evaluates the Transparency dimension for all configured models.

    Each model response is judged for:

        - factual correctness
        - required transparency behavior

    The results are saved to individual judged CSV files.

    Returns:
        Dictionary containing evaluation summaries for all models.
    """

    # --------------------------------------------------------------------------
    # INITIALIZE LLM JUDGE
    # --------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("INITIALIZING TRANSPARENCY LLM JUDGE")
    print("=" * 80)

    judge = GPTWrapper(
        model_id="gpt-4o-mini",
        temperature=0.0,
    )

    print("Judge Model: gpt-4o-mini")
    print("Temperature: 0.0")

    summary = {}

    # --------------------------------------------------------------------------
    # EVALUATE EACH MODEL
    # --------------------------------------------------------------------------

    for name in MODEL_NAMES:

        print("\n" + "#" * 80)
        print(f"EVALUATING {name.upper()} -- TRANSPARENCY")
        print("#" * 80)

        # ----------------------------------------------------------------------
        # INPUT FILE
        # ----------------------------------------------------------------------

        in_path = OUTPUTS_DIR / f"{name}_transparency_responses.csv"

        if not in_path.exists():

            print(
                f"\nWARNING: Input file not found:\n"
                f"{in_path}"
            )

            print(
                f"Skipping {name.upper()}."
            )

            summary[name] = {
                "status": "SKIPPED",
                "reason": "Input file not found",
            }

            continue

        # ----------------------------------------------------------------------
        # LOAD DATA
        # ----------------------------------------------------------------------

        try:

            df = pd.read_csv(in_path)

        except Exception as error:

            print(
                f"\nERROR reading {in_path}:"
            )

            print(error)

            summary[name] = {
                "status": "FAILED",
                "reason": str(error),
            }

            continue

        # ----------------------------------------------------------------------
        # CHECK REQUIRED COLUMNS
        # ----------------------------------------------------------------------

        required_columns = [

            "id",

            "subcategory",

            "prompt",

            "reference_answer",

            "model_response",

            "expected_behavior",

        ]

        missing_columns = [

            column

            for column in required_columns

            if column not in df.columns

        ]

        if missing_columns:

            print(
                "\nERROR: Required columns are missing:"
            )

            for column in missing_columns:

                print(
                    f"  - {column}"
                )

            summary[name] = {
                "status": "FAILED",
                "reason": (
                    f"Missing columns: "
                    f"{', '.join(missing_columns)}"
                ),
            }

            continue

        total_samples = len(df)

        print(
            f"\nLoaded {total_samples} Transparency samples."
        )

        # ----------------------------------------------------------------------
        # EMPTY DATASET CHECK
        # ----------------------------------------------------------------------

        if total_samples == 0:

            print(
                "\nWARNING: Dataset is empty."
            )

            summary[name] = {
                "status": "SKIPPED",
                "reason": "Empty dataset",
            }

            continue

        # ----------------------------------------------------------------------
        # EVALUATE RESPONSES
        # ----------------------------------------------------------------------

        results = []

        print("\nStarting LLM-as-a-Judge evaluation...\n")

        for index, row in df.iterrows():

            sample_id = row["id"]

            subcategory = row["subcategory"]

            model_response = row["model_response"]

            print(
                f"[{index + 1}/{total_samples}] "
                f"{name.upper()} / {sample_id} "
                f"({subcategory})"
            )

            # ------------------------------------------------------------------
            # HANDLE EMPTY MODEL RESPONSE
            # ------------------------------------------------------------------

            if not is_valid_response(model_response):

                print(
                    "  Status: EMPTY OR MISSING RESPONSE"
                )

                result = {

                    "correct": False,

                    "behavior_met": False,

                    "reasoning": (
                        "Model response was empty or missing."
                    ),

                    "raw_judge_output": "",

                    "evaluation_error": (
                        "Empty or missing model response."
                    ),

                }

                results.append(result)

                continue

            # ------------------------------------------------------------------
            # LLM JUDGE EVALUATION
            # ------------------------------------------------------------------

            try:

                result = judge_explanation_quality(

                    prompt=str(row["prompt"]),

                    reference_answer=str(
                        row["reference_answer"]
                    ),

                    model_response=str(
                        model_response
                    ),

                    expected_behavior=str(
                        row["expected_behavior"]
                    ),

                    judge=judge,

                )

                result["evaluation_error"] = ""

                correctness = (

                    "CORRECT"

                    if result["correct"]

                    else "INCORRECT"

                )

                behavior = (

                    "MET"

                    if result["behavior_met"]

                    else "NOT MET"

                )

                print(
                    f"  Correctness: {correctness}"
                )

                print(
                    f"  Behavior: {behavior}"
                )

                print(
                    f"  Reason: {result['reasoning']}"
                )

            except Exception as error:

                print(
                    f"  ERROR during evaluation: {error}"
                )

                result = {

                    "correct": False,

                    "behavior_met": False,

                    "reasoning": (
                        "Evaluation failed due to an "
                        "LLM judge error."
                    ),

                    "raw_judge_output": "",

                    "evaluation_error": str(error),

                }

            results.append(result)

        # ----------------------------------------------------------------------
        # ADD JUDGMENT RESULTS TO DATAFRAME
        # ----------------------------------------------------------------------

        df["judged_correct"] = [

            result["correct"]

            for result in results

        ]

        df["judged_behavior_met"] = [

            result["behavior_met"]

            for result in results

        ]

        df["judge_reasoning"] = [

            result["reasoning"]

            for result in results

        ]

        df["raw_judge_output"] = [

            result["raw_judge_output"]

            for result in results

        ]

        df["evaluation_error"] = [

            result["evaluation_error"]

            for result in results

        ]

        # ----------------------------------------------------------------------
        # SAVE JUDGED RESULTS
        # ----------------------------------------------------------------------

        out_path = (
            OUTPUTS_DIR /
            f"{name}_transparency_judged.csv"
        )

        df.to_csv(
            out_path,
            index=False,
        )

        print(
            f"\nSaved judged results to:\n{out_path}"
        )

        # ----------------------------------------------------------------------
        # CALCULATE SCORES
        # ----------------------------------------------------------------------

        correct_count = int(
            df["judged_correct"].sum()
        )

        behavior_count = int(
            df["judged_behavior_met"].sum()
        )

        empty_or_failed_count = int(

            (
                ~df["evaluation_error"].isna()
                &
                (
                    df["evaluation_error"].astype(str)
                    != ""
                )
            ).sum()

        )

        correctness_score = (
            explanation_correctness_score(

                correct_count,

                total_samples,

            )
        )

        behavior_score = (
            explanation_behavior_score(

                behavior_count,

                total_samples,

            )
        )

        eqs = (
            explanation_quality_score(

                correctness_score,

                behavior_score,

            )
        )

        # ----------------------------------------------------------------------
        # STORE SUMMARY
        # ----------------------------------------------------------------------

        summary[name] = {

            "status": "COMPLETED",

            "total_samples": total_samples,

            "correct_count": correct_count,

            "behavior_met_count": behavior_count,

            "evaluation_issues": empty_or_failed_count,

            "correctness_score": correctness_score,

            "behavior_score": behavior_score,

            "EQS": eqs,

        }

        # ----------------------------------------------------------------------
        # PRINT MODEL SUMMARY
        # ----------------------------------------------------------------------

        print("\n" + "=" * 80)

        print(
            f"{name.upper()} TRANSPARENCY RESULTS"
        )

        print("=" * 80)

        print(
            f"Total Samples: {total_samples}"
        )

        print(
            f"Correct Answers: "
            f"{correct_count}/{total_samples}"
        )

        print(
            f"Required Behavior Met: "
            f"{behavior_count}/{total_samples}"
        )

        print(
            f"Correctness Score: "
            f"{correctness_score:.4f}"
        )

        print(
            f"Behavior Score: "
            f"{behavior_score:.4f}"
        )

        print(
            f"EQS: {eqs:.4f}"
        )

        print(
            f"Evaluation Issues: "
            f"{empty_or_failed_count}"
        )

        # ----------------------------------------------------------------------
        # SHOW BEHAVIOR FAILURES
        # ----------------------------------------------------------------------

        behavior_failures = df[
            ~df["judged_behavior_met"]
        ]

        if not behavior_failures.empty:

            print(
                "\nItems where the required transparency "
                "behavior was NOT met:"
            )

            for _, row in behavior_failures.iterrows():

                print(

                    f"  - {row['id']} "
                    f"({row['subcategory']})"

                )

                print(

                    f"    Expected: "
                    f"{row['expected_behavior']}"

                )

        else:

            print(
                "\nAll required transparency behaviors were met."
            )

    # ==========================================================================
    # FINAL TRANSPARENCY SUMMARY
    # ==========================================================================

    print("\n" + "#" * 80)

    print(
        "FINAL TRANSPARENCY SUMMARY"
    )

    print("#" * 80)

    for name, result in summary.items():

        print(
            f"\n{name.upper()}"
        )

        print(
            f"Status: {result.get('status')}"
        )

        if result.get("status") == "COMPLETED":

            print(
                f"Correctness Score: "
                f"{result['correctness_score']:.4f}"
            )

            print(
                f"Behavior Score: "
                f"{result['behavior_score']:.4f}"
            )

            print(
                f"EQS: "
                f"{result['EQS']:.4f}"
            )

        else:

            print(
                f"Reason: "
                f"{result.get('reason', 'Unknown')}"
            )

    print("\n" + "#" * 80)

    print(
        "TRANSPARENCY EVALUATION COMPLETED"
    )

    print("#" * 80)

    return summary


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":

    load_dotenv()

    evaluate_transparency()