"""
Stage 2 for the Bias dimension.

Reads outputs/{model}_bias_responses.csv (produced by main.py), judges
each response against its reference_answer, and computes the
Demographic Output Parity Score (DOPS) per model from two components:

- ambiguous accuracy: did the model correctly refuse to assume a group
  when the facts given do not distinguish them?
- explicit-info accuracy: did the model correctly follow the facts once
  they were made explicit?

Also reports which specific assumption (if any) the model made on each
incorrect ambiguous item, providing qualitative evidence in addition
to the overall DOPS score.
"""

import os
import pandas as pd

from dotenv import load_dotenv

from evaluators.parity_checker import judge_bias_response

from metrics.calculator import (
    ambiguous_bias_avoidance,
    explicit_info_accuracy,
    demographic_output_parity_score,
)

from models.gpt_wrapper import GPTWrapper


# ============================================================
# MODELS TO EVALUATE
# ============================================================

MODEL_NAMES = [
    "gpt",
    "llama",
]


# ============================================================
# BIAS EVALUATION
# ============================================================

def evaluate_bias():

    # --------------------------------------------------------
    # Initialize LLM-as-a-Judge
    # --------------------------------------------------------

    judge = GPTWrapper(
        model_id="gpt-4o-mini",
        temperature=0.0,
    )

    summary = {}

    # ========================================================
    # EVALUATE EACH MODEL
    # ========================================================

    for name in MODEL_NAMES:

        print("\n" + "=" * 50)
        print(f"Evaluating {name.upper()} -- Bias")
        print("=" * 50)

        in_path = f"outputs/{name}_bias_responses.csv"

        # ----------------------------------------------------
        # Skip missing output files
        # ----------------------------------------------------

        if not os.path.exists(in_path):

            print(
                f"  SKIPPED -- response file not found:\n"
                f"  {in_path}"
            )

            continue

        # ----------------------------------------------------
        # Load responses
        # ----------------------------------------------------

        df = pd.read_csv(in_path)

        if df.empty:

            print(
                f"  SKIPPED -- response file is empty."
            )

            continue

        # ----------------------------------------------------
        # Validate required columns
        # ----------------------------------------------------

        required_columns = [
            "id",
            "prompt",
            "reference_answer",
            "model_response",
            "variant_label",
            "subcategory",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            print(
                f"  SKIPPED -- missing required columns: "
                f"{missing_columns}"
            )

            continue

        # ----------------------------------------------------
        # Judge each response
        # ----------------------------------------------------

        results = []

        for _, row in df.iterrows():

            result = judge_bias_response(

                prompt=row["prompt"],

                reference_answer=row["reference_answer"],

                model_response=row["model_response"],

                judge=judge,

                variant_label=row["variant_label"],

                expected_behavior=row["expected_behavior"],

            )

            results.append(result)

            correctness = (
                "CORRECT"
                if result["correct"]
                else "INCORRECT"
            )

            group_note = ""

            if result["assumed_group"] != "NONE":

                group_note = (
                    f" (assumed: "
                    f"{result['assumed_group']})"
                )

            print(
                f"  {name}/{row['id']} "
                f"({row['variant_label']}, "
                f"{row['subcategory']}): "
                f"{correctness}"
                f"{group_note} -- "
                f"{result['reasoning']}"
            )

        # ----------------------------------------------------
        # Add judge results to DataFrame
        # ----------------------------------------------------

        df["judged_correct"] = [
            result["correct"]
            for result in results
        ]

        df["assumed_group"] = [
            result["assumed_group"]
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

        # ----------------------------------------------------
        # Save judged results
        # ----------------------------------------------------

        out_path = (
            f"outputs/{name}_bias_judged.csv"
        )

        df.to_csv(
            out_path,
            index=False,
            encoding="utf-8",
        )

        print(
            f"\nSaved judged results to {out_path}"
        )

        # ====================================================
        # SPLIT AMBIGUOUS / DISAMBIGUATED ITEMS
        # ====================================================

        ambiguous_rows = df[
            df["variant_label"] == "ambiguous"
        ]

        disambiguated_rows = df[
            df["variant_label"] == "disambiguated"
        ]

        # ----------------------------------------------------
        # Calculate ambiguous bias avoidance
        # ----------------------------------------------------

        amb_correct = int(
            ambiguous_rows["judged_correct"].sum()
        )

        amb_total = len(
            ambiguous_rows
        )

        if amb_total > 0:

            amb_score = ambiguous_bias_avoidance(
                amb_correct,
                amb_total,
            )

        else:

            amb_score = 0.0

        # ----------------------------------------------------
        # Calculate explicit-information accuracy
        # ----------------------------------------------------

        exp_correct = int(
            disambiguated_rows["judged_correct"].sum()
        )

        exp_total = len(
            disambiguated_rows
        )

        if exp_total > 0:

            exp_score = explicit_info_accuracy(
                exp_correct,
                exp_total,
            )

        else:

            exp_score = 0.0

        # ----------------------------------------------------
        # Calculate final DOPS
        # ----------------------------------------------------

        dops = demographic_output_parity_score(
            amb_score,
            exp_score,
        )

        # ====================================================
        # FIND ASSUMPTIONS ON AMBIGUOUS ITEMS
        # ====================================================

        stereotype_hits = ambiguous_rows[

            (~ambiguous_rows["judged_correct"])

            &

            (
                ambiguous_rows["assumed_group"]
                != "NONE"
            )

        ]

        # ----------------------------------------------------
        # Save summary
        # ----------------------------------------------------

        summary[name] = {

            "DOPS": dops,

            "ambiguous_score": amb_score,

            "explicit_score": exp_score,

        }

        # ====================================================
        # PRINT MODEL RESULTS
        # ====================================================

        print(

            f"\n{name.upper()}: "

            f"DOPS = {dops:.2f}\n"

            f"  Ambiguous Bias Avoidance: "
            f"{amb_correct}/{amb_total} "
            f"({amb_score:.2f})\n"

            f"  Explicit Information Accuracy: "
            f"{exp_correct}/{exp_total} "
            f"({exp_score:.2f})"

        )

        # ----------------------------------------------------
        # Print qualitative bias findings
        # ----------------------------------------------------

        if not stereotype_hits.empty:

            print(
                "\n  Assumptions made on ambiguous items:"
            )

            for _, result_row in stereotype_hits.iterrows():

                print(
                    f"    - "
                    f"{result_row['subcategory']}: "
                    f"assumed "
                    f"{result_row['assumed_group']}"
                )

        else:

            print(
                "\n  No specific group assumptions detected "
                "on incorrect ambiguous items."
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 50)

    print("BIAS SUMMARY")

    print("=" * 50)

    if not summary:

        print(
            "No models were successfully evaluated."
        )

    else:

        for name, scores in summary.items():

            print(

                f"{name.upper()}: "

                f"DOPS = {scores['DOPS']:.2f} | "

                f"Ambiguous = "
                f"{scores['ambiguous_score']:.2f} | "

                f"Explicit = "
                f"{scores['explicit_score']:.2f}"

            )

    print("=" * 50)

    return summary


# ============================================================
# RUN EVALUATOR
# ============================================================

if __name__ == "__main__":

    load_dotenv()

    evaluate_bias()