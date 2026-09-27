import pandas as pd


def load_audit_dataset(path: str) -> pd.DataFrame:
    """Loads unified_audit_dataset_v2.csv and does light validation."""
    df = pd.read_csv(path)
    required_cols = {"id", "dimension", "metric", "prompt"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def get_items_for_dimension(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Returns just the rows for one audit dimension, e.g. 'truthfulness'."""
    subset = df[df["dimension"] == dimension].copy()
    if subset.empty:
        raise ValueError(f"No rows found for dimension='{dimension}'")
    return subset
