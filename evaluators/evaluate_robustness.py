"""
Stage 2 for the Robustness dimension.

Reads outputs/{model}_robustness_responses.csv, pairs base and
perturbed prompts using pair_id, evaluates semantic consistency or
prompt-injection resistance, and calculates the Robustness Score (RS).
"""

import os
import pandas as pd

from dotenv import load_dotenv

from evaluators.similarity_checker import (
    judge_consistency,
    judge_injection_bypass,
)

from metrics.calculator import (
    semantic_consistency_score,
    injection_bypass_rate,
    robustness_score,
)

from models.gpt_wrapper import GPTWrapper


MODEL_NAMES = ["gemini", "gpt", "llama"]


def evaluate_robustness():

    judge = GPTWrapper(
        model_id="gpt-4o-mini",
        temperature=0.0,
    )

    summary = {}

    for name in MODEL_NAMES:

        print("\n" + "=" * 50)
        print(f"Evaluating {name.upper()} — Robustness")
        print("=" * 50)

        in_path = f"outputs/{name}_robustness_responses.csv"

        # Skip models whose response file does not exist.
        if not os.path.exists(in_path):

            print(
                f"  SKIPPED — response file not found:\n"
                f"  {in_path}"
            )

            continue

        df = pd.read_csv(in_path)

        if df.empty:

            print(
                f"  SKIPPED — response file is empty."
            )

            continue

        results = []

        # ====================================================
        # EVALUATE EACH BASE/PERTURBED PAIR
        # ====================================================

        for pair_id, group in df.groupby("pair_id"):

            base_rows = group[
                group["variant_label"] == "base"
            ]

            perturbed_rows = group[
                group["variant_label"] == "perturbed"
            ]

            # Skip incomplete pairs.
            if base_rows.empty or perturbed_rows.empty:

                print(
                    f"  {pair_id}: SKIPPED — incomplete pair "
                    f"(base or perturbed response missing)"
                )

                continue

            base_row = base_rows.iloc[0]
            perturbed_row = perturbed_rows.iloc[0]

            subcategory = base_row["subcategory"]

            # ================================================
            # PROMPT INJECTION PAIR
            # ================================================

            if subcategory == "prompt_injection":

                result = judge_injection_bypass(

                    base_prompt=base_row["prompt"],
                    base_response=base_row["model_response"],

                    perturbed_prompt=perturbed_row["prompt"],
                    perturbed_response=perturbed_row["model_response"],

                    judge=judge,
                )

                result.update({

                    "pair_id": pair_id,
                    "subcategory": subcategory,
                    "pair_type": "injection",

                })

                label = (
                    "BYPASSED"
                    if result["bypassed"]
                    else "RESISTED"
                )

            # ================================================
            # NOISE / PARAPHRASE PAIRS
            # ================================================

            else:

                result = judge_consistency(

                    base_prompt=base_row["prompt"],
                    base_response=base_row["model_response"],

                    perturbed_prompt=perturbed_row["prompt"],
                    perturbed_response=perturbed_row["model_response"],

                    judge=judge,
                )

                result.update({

                    "pair_id": pair_id,
                    "subcategory": subcategory,
                    "pair_type": "consistency",

                })

                label = (
                    "CONSISTENT"
                    if result["consistent"]
                    else "INCONSISTENT"
                )

            results.append(result)

            print(
                f"  {name}/{pair_id} ({subcategory}): "
                f"{label} — {result['reasoning']}"
            )

        # ====================================================
        # HANDLE NO RESULTS
        # ====================================================

        if not results:

            print(
                f"\nNo complete pairs available for "
                f"{name.upper()}."
            )

            continue

        # ====================================================
        # SAVE JUDGED RESULTS
        # ====================================================

        results_df = pd.DataFrame(results)

        out_path = (
            f"outputs/{name}_robustness_judged.csv"
        )

        results_df.to_csv(
            out_path,
            index=False,
            encoding="utf-8",
        )

        print(
            f"\nSaved judged results to {out_path}"
        )

        # ====================================================
        # CALCULATE METRICS
        # ====================================================

        consistency_rows = results_df[
            results_df["pair_type"] == "consistency"
        ]

        injection_rows = results_df[
            results_df["pair_type"] == "injection"
        ]

        # --------------------------------------------
        # Consistency score
        # --------------------------------------------

        total_consistency_pairs = len(consistency_rows)

        consistent_count = (
            int(consistency_rows["consistent"].sum())
            if total_consistency_pairs > 0
            else 0
        )

        if total_consistency_pairs > 0:

            consistency_score = semantic_consistency_score(

                consistent_count,
                total_consistency_pairs,

            )

        else:

            consistency_score = 0.0

        # --------------------------------------------
        # Injection bypass rate
        # --------------------------------------------

        total_injection_pairs = len(injection_rows)

        bypassed_count = (

            int(injection_rows["bypassed"].sum())

            if total_injection_pairs > 0

            else 0

        )

        if total_injection_pairs > 0:

            bypass_rate = injection_bypass_rate(

                bypassed_count,
                total_injection_pairs,

            )

        else:

            bypass_rate = 0.0

        # --------------------------------------------
        # Final robustness score
        # --------------------------------------------

        rs = robustness_score(

            consistency_score,

            bypass_rate,

        )

        summary[name] = {

            "RS": rs,

            "consistency_score": consistency_score,

            "injection_bypass_rate": bypass_rate,

            "consistent_pairs": consistent_count,

            "total_consistency_pairs": total_consistency_pairs,

            "bypassed_injections": bypassed_count,

            "total_injection_pairs": total_injection_pairs,

        }

        # ====================================================
        # PRINT MODEL RESULT
        # ====================================================

        print(
            f"\n{name.upper()}: RS = {rs:.2f}"
        )

        print(
            f"  Consistency: "
            f"{consistent_count}/{total_consistency_pairs}"
        )

        print(
            f"  Injection bypassed: "
            f"{bypassed_count}/{total_injection_pairs}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 50)
    print("ROBUSTNESS SUMMARY")
    print("=" * 50)

    for name, scores in summary.items():

        print(

            f"{name.upper()}: "
            f"RS = {scores['RS']:.2f} | "
            f"Consistency = {scores['consistency_score']:.2f} | "
            f"Injection Bypass = "
            f"{scores['injection_bypass_rate']:.2f}"

        )

    print("=" * 50)

    return summary


if __name__ == "__main__":

    load_dotenv()

    evaluate_robustness()