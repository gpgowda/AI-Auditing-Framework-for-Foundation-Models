"""
Stage 2 for the Safety dimension: Evaluate Responses -> Calculate Audit Metrics.

Reads:

    outputs/{model}_safety_responses.csv

produced by main.py, judges each model response as either:

    REFUSED
    COMPLIED

and calculates Harm Refusal Rate (HRR).

The Safety dataset contains two types of prompts:

- hazardous:
    The model is expected to refuse the harmful request.

- benign:
    The model is expected to answer normally and should NOT refuse.

The Harm Refusal Rate therefore considers both:

1. Refusal of hazardous requests.
2. Over-refusal of benign requests.
"""

import pandas as pd
from dotenv import load_dotenv

from evaluators.safety_checker import judge_refusal
from metrics.calculator import harm_refusal_rate
from models.gpt_wrapper import GPTWrapper


# Claude excluded — see config/settings.yaml
MODEL_NAMES = ["gemini", "gpt", "llama"]


def evaluate_safety():
    """
    Evaluates the Safety responses for all audited models.

    For each model:

    1. Loads the generated Safety responses.
    2. Uses an independent judge model to classify each response.
    3. Saves the judged results.
    4. Calculates Harm Refusal Rate (HRR).
    """

    # Use a separate judge model rather than the model being evaluated.
    # Temperature 0.0 improves evaluation consistency.
    judge = GPTWrapper(
        model_id="gpt-4o-mini",
        temperature=0.0
    )

    summary = {}

    # ==========================================================
    # Evaluate each audited model
    # ==========================================================

    for name in MODEL_NAMES:

        print("\n" + "=" * 50)
        print(f"Evaluating {name.upper()} — Safety")
        print("=" * 50)

        # ------------------------------------------------------
        # Load generated Safety responses
        # ------------------------------------------------------

        in_path = f"outputs/{name}_safety_responses.csv"

        df = pd.read_csv(in_path)

        # Check that the required dataset columns exist.
        required_columns = [
            "id",
            "prompt",
            "model_response",
            "variant_label",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{in_path} is missing required columns: "
                f"{missing_columns}"
            )

        results = []

        # ------------------------------------------------------
        # Judge every response
        # ------------------------------------------------------

        for _, row in df.iterrows():

            result = judge_refusal(
                prompt=row["prompt"],
                model_response=row["model_response"],
                judge=judge,
            )

            results.append(result)

            # Get the explicit verdict returned by the judge.
            behavior = result["behavior"]

            # Detect malformed judge outputs instead of silently
            # treating them as COMPLIED.
            if behavior not in ["REFUSED", "COMPLIED"]:

                print(
                    f"  WARNING: {name}/{row['id']} produced "
                    f"an unexpected judge verdict: {behavior!r}"
                )

                print(
                    f"  Raw judge output: "
                    f"{result['raw_judge_output']}"
                )

            variant = row["variant_label"]

            print(
                f"  {name}/{row['id']} ({variant}): "
                f"{behavior} — {result['reasoning']}"
            )

        # ------------------------------------------------------
        # Save judge results into the DataFrame
        # ------------------------------------------------------

        df["judged_behavior"] = [
            r["behavior"]
            for r in results
        ]

        df["judged_refused"] = [
            r["refused"]
            for r in results
        ]

        df["judge_reasoning"] = [
            r["reasoning"]
            for r in results
        ]

        df["raw_judge_output"] = [
            r["raw_judge_output"]
            for r in results
        ]

        # ------------------------------------------------------
        # Save judged results
        # ------------------------------------------------------

        out_path = f"outputs/{name}_safety_judged.csv"

        df.to_csv(
            out_path,
            index=False
        )

        print(
            f"\nSaved judged results to {out_path}"
        )

        # ------------------------------------------------------
        # Separate hazardous and benign prompts
        # ------------------------------------------------------

        hazardous = df[
            df["variant_label"].str.lower() == "hazardous"
        ]

        benign = df[
            df["variant_label"].str.lower() == "benign"
        ]

        # ------------------------------------------------------
        # Calculate Safety counts
        # ------------------------------------------------------

        # Correct behavior:
        # Hazardous prompts should be refused.
        hazardous_refused = int(
            hazardous["judged_refused"].sum()
        )

        total_hazardous = len(hazardous)

        # Incorrect behavior:
        # Benign prompts should NOT be refused.
        benign_false_refused = int(
            benign["judged_refused"].sum()
        )

        total_benign = len(benign)

        # ------------------------------------------------------
        # Validate dataset split
        # ------------------------------------------------------

        if total_hazardous == 0:

            raise ValueError(
                f"No hazardous rows found in {in_path}. "
                f"Check the variant_label column."
            )

        if total_benign == 0:

            raise ValueError(
                f"No benign rows found in {in_path}. "
                f"Check the variant_label column."
            )

        # ------------------------------------------------------
        # Calculate Harm Refusal Rate
        # ------------------------------------------------------

        hrr = harm_refusal_rate(
            hazardous_refused=hazardous_refused,
            total_hazardous=total_hazardous,
            benign_false_refused=benign_false_refused,
            total_benign=total_benign,
        )

        # Store results for the final summary.
        summary[name] = {

            "HRR": hrr,

            "hazardous_refused": hazardous_refused,

            "total_hazardous": total_hazardous,

            "benign_false_refused": benign_false_refused,

            "total_benign": total_benign,
        }

        # ------------------------------------------------------
        # Print model results
        # ------------------------------------------------------

        print()

        print(
            f"{name.upper()}: HRR = {hrr:.2f}"
        )

        print(
            f"  Hazardous prompts refused: "
            f"{hazardous_refused}/{total_hazardous}"
        )

        print(
            f"  Benign prompts wrongly refused: "
            f"{benign_false_refused}/{total_benign}"
        )

    # ==========================================================
    # Final Safety Summary
    # ==========================================================

    print("\n" + "=" * 50)
    print("SAFETY SUMMARY")
    print("=" * 50)

    for name, scores in summary.items():

        print(
            f"{name.upper()}: "
            f"HRR = {scores['HRR']:.2f} | "
            f"Hazardous Refusal = "
            f"{scores['hazardous_refused']}/"
            f"{scores['total_hazardous']} | "
            f"Benign False Refusal = "
            f"{scores['benign_false_refused']}/"
            f"{scores['total_benign']}"
        )

    print("=" * 50)

    return summary


if __name__ == "__main__":

    load_dotenv()

    evaluate_safety()