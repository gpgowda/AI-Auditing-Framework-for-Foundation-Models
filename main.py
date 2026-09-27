"""
Generation stage of the AI Auditing Framework pipeline.

Runs each enabled audit dimension against every enabled model
and saves the raw responses.

The pipeline is generalized so that adding a new audit dimension
(truthfulness, safety, robustness, bias, privacy, etc.) only requires
adding the dimension to dimensions_enabled in config/settings.yaml.
"""

import os
import time

import yaml
import pandas as pd

from dotenv import load_dotenv

from utils.data_loader import load_audit_dataset, get_items_for_dimension
from prompts.prompt_builder import build_prompt

from models.gpt_wrapper import GPTWrapper
from models.gemini_wrapper import GeminiWrapper
from models.llama_local_wrapper import LlamaLocalWrapper


# ============================================================
# MODEL PROVIDER MAP
# ============================================================

# Maps the provider name in settings.yaml
# to the corresponding Python wrapper class.

PROVIDER_MAP = {
    "gpt": GPTWrapper,
    "gemini": GeminiWrapper,
    "llama_local": LlamaLocalWrapper,
}


# ============================================================
# CONFIGURATION LOADING
# ============================================================

def load_config(path: str = "config/settings.yaml") -> dict:
    """
    Load the YAML configuration file.

    Parameters
    ----------
    path : str
        Path to the settings.yaml configuration file.

    Returns
    -------
    dict
        Configuration dictionary.
    """

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


# ============================================================
# MODEL INITIALIZATION
# ============================================================

def create_model(model_cfg: dict, request_cfg: dict):
    """
    Create the correct model wrapper based on the provider
    specified in settings.yaml.

    Parameters
    ----------
    model_cfg : dict
        Configuration for one model.

    request_cfg : dict
        Request configuration settings.

    Returns
    -------
    object
        Initialized model wrapper.
    """

    provider = model_cfg.get("provider")

    if provider not in PROVIDER_MAP:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Available providers: {list(PROVIDER_MAP.keys())}"
        )

    wrapper_cls = PROVIDER_MAP[provider]

    model_id = model_cfg.get("model_id")

    # Provider-specific options (e.g. Gemini's thinking_level).
    extra_kwargs = {}

    if "thinking_level" in model_cfg:
        extra_kwargs["thinking_level"] = model_cfg["thinking_level"]

    return wrapper_cls(
        model_id=model_id,
        temperature=request_cfg.get("temperature", 0.0),
        max_tokens=request_cfg.get("max_output_tokens", 300),
        **extra_kwargs,
    )


# ============================================================
# DIMENSION GENERATION
# ============================================================
def run_dimension_slice(config: dict, dimension: str):
    """
    Run all enabled models on all dataset items belonging
    to one audit dimension.

    Previously generated responses are automatically detected
    and skipped, allowing the pipeline to resume safely after
    API quota limits or interruptions.
    """

    # ========================================================
    # LOAD DATASET
    # ========================================================

    df = load_audit_dataset(config["dataset_path"])

    items = get_items_for_dimension(df, dimension)

    if items.empty:

        print("\n" + "=" * 60)
        print(
            f"WARNING: No dataset items found "
            f"for dimension '{dimension}'"
        )
        print("=" * 60)

        return {}

    # ========================================================
    # ENSURE OUTPUT DIRECTORY EXISTS
    # ========================================================

    os.makedirs("outputs", exist_ok=True)

    # ========================================================
    # PRINT DIMENSION HEADER
    # ========================================================

    print("\n" + "#" * 60)
    print(
        f"# DIMENSION: {dimension.upper()} "
        f"({len(items)} prompts)"
    )
    print("#" * 60)

    generation_summary = {}

    # ========================================================
    # RUN EACH MODEL
    # ========================================================

    for name, model_cfg in config["models"].items():

        # ----------------------------------------------------
        # SKIP DISABLED MODELS
        # ----------------------------------------------------

        if not model_cfg.get("enabled", False):

            print(
                f"\nSkipping {name} "
                f"(disabled in settings.yaml)"
            )

            generation_summary[name] = 0
            continue

        provider = model_cfg.get("provider")
        model_id = model_cfg.get("model_id")

        # ----------------------------------------------------
        # VALIDATE PROVIDER
        # ----------------------------------------------------

        if provider not in PROVIDER_MAP:

            print(
                f"\nSkipping {name.upper()}: "
                f"unknown provider '{provider}'"
            )

            generation_summary[name] = 0
            continue

        # ----------------------------------------------------
        # MODEL HEADER
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print(
            f"Running {name.upper()} "
            f"({model_id}) — {dimension.upper()}"
        )
        print("=" * 60)

        # ====================================================
        # OUTPUT FILE PATH
        # ====================================================

        out_path = (
            f"outputs/{name}_{dimension}_responses.csv"
        )

        # ====================================================
        # LOAD EXISTING RESPONSES
        # ====================================================

        completed_ids = set()

        if os.path.exists(out_path):

            try:

                existing_df = pd.read_csv(out_path)

                if "id" in existing_df.columns:

                    completed_ids = set(
                        existing_df["id"]
                        .dropna()
                        .astype(str)
                    )

                print(
                    f"\nFound {len(completed_ids)} "
                    f"previously completed responses."
                )

            except Exception as error:

                print(
                    f"\nWARNING: Could not read existing "
                    f"output file: {error}"
                )

        # ====================================================
        # CREATE MODEL WRAPPER
        # ====================================================

        try:

            model = create_model(
                model_cfg=model_cfg,
                request_cfg=config.get("request", {}),
            )

        except KeyError as error:

            print(
                f"\n{name.upper()} SKIPPED — "
                f"missing configuration or API key: {error}"
            )

            generation_summary[name] = 0
            continue

        except Exception as error:

            print(
                f"\n{name.upper()} SKIPPED — "
                f"could not initialize model: {error}"
            )

            generation_summary[name] = 0
            continue

        # ====================================================
        # GENERATION COUNTERS
        # ====================================================

        newly_generated = 0
        skipped = 0

        # ====================================================
        # RUN EACH DATASET ITEM
        # ====================================================

        for _, item in items.iterrows():

            item_id = str(item["id"])

            # ------------------------------------------------
            # SKIP ALREADY COMPLETED ITEMS
            # ------------------------------------------------

            if item_id in completed_ids:

                print(
                    f"  ⏭ {item_id} already completed — skipping"
                )

                skipped += 1
                continue

            try:

                # ------------------------------------------------
                # BUILD PROMPT
                # ------------------------------------------------

                full_prompt = build_prompt(item["prompt"])

                # ------------------------------------------------
                # SEND REQUEST TO MODEL
                # ------------------------------------------------

                response = model.query(full_prompt)

                # ------------------------------------------------
                # PRESERVE DATASET ROW
                # ------------------------------------------------

                row = item.to_dict()

                row["model_response"] = response.text
                row["model_id"] = response.model_id
                row["audit_model"] = name

                # ------------------------------------------------
                # SAVE IMMEDIATELY
                # ------------------------------------------------

                row_df = pd.DataFrame([row])

                file_exists = os.path.exists(out_path)

                row_df.to_csv(
                    out_path,
                    mode="a" if file_exists else "w",
                    header=not file_exists,
                    index=False,
                    encoding="utf-8",
                )

                # Mark as completed.
                completed_ids.add(item_id)

                newly_generated += 1

                print(
                    f"  ✓ {item_id} completed "
                    f"and saved"
                )

                # =================================================
                # GEMINI RATE LIMIT DELAY
                # =================================================

                if name == "gemini":

                    delay = config.get(
                        "gemini_delay_seconds",
                        15,
                    )

                    print(
                        f"    Waiting {delay} seconds "
                        f"before the next Gemini request..."
                    )

                    time.sleep(delay)

            except Exception as error:

                print(
                    f"  ✗ {item_id} FAILED: {error}"
                )

                # =================================================
                # GEMINI FAILURE DELAY
                # =================================================

                if name == "gemini":

                    failure_delay = config.get(
                        "gemini_failure_delay_seconds",
                        40,
                    )

                    print(
                        f"    Gemini request failed. "
                        f"Waiting {failure_delay} seconds..."
                    )

                    time.sleep(failure_delay)

        # ====================================================
        # MODEL SUMMARY
        # ====================================================

        print("\n" + "-" * 60)

        print(
            f"{name.upper()} — {dimension.upper()} SUMMARY"
        )

        print(
            f"Newly generated: {newly_generated}"
        )

        print(
            f"Already completed: {skipped}"
        )

        print(
            f"Output file: {out_path}"
        )

        print("-" * 60)

        generation_summary[name] = newly_generated

    return generation_summary

# ============================================================
# MAIN PIPELINE
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # LOAD ENVIRONMENT VARIABLES
    # ========================================================

    load_dotenv()

    # ========================================================
    # LOAD CONFIGURATION
    # ========================================================

    config = load_config()

    # ========================================================
    # PIPELINE HEADER
    # ========================================================

    print("\n" + "=" * 60)
    print("AI AUDITING FRAMEWORK")
    print("RESPONSE GENERATION PIPELINE")
    print("=" * 60)

    # ========================================================
    # RUN ENABLED DIMENSIONS
    # ========================================================

    dimensions = config.get(
        "dimensions_enabled",
        [],
    )

    if not dimensions:

        print(
            "\nWARNING: No audit dimensions are enabled "
            "in settings.yaml"
        )

    all_summaries = {}

    for dimension in dimensions:

        all_summaries[dimension] = run_dimension_slice(
            config,
            dimension,
        )

    # ========================================================
    # GENERATION SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)

    for dimension, summary in all_summaries.items():

        print(f"\n{dimension.capitalize()} responses:")

        if not summary:

            print("  No responses generated.")

        else:

            for model, count in summary.items():

                print(
                    f"  {model.upper()}: {count}"
                )

    # ========================================================
    # NEXT STEP
    # ========================================================

    print("\n" + "=" * 60)
    print("NEXT STEP")
    print("=" * 60)

    for dimension in dimensions:

        print(
            f"\nRun the {dimension.capitalize()} evaluator:"
        )

        print(
            f"  python -m evaluators.evaluate_{dimension}"
        )

    print("\n" + "=" * 60)