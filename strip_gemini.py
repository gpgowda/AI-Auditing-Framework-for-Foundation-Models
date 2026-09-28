import pandas as pd

path = "reports/human_annotation_sample.csv"
df = pd.read_csv(path, dtype=str)

before = len(df)
df = df[~df["validation_key"].str.startswith("gemini")]
after = len(df)

df.to_csv(path, index=False)
print(f"Removed {before - after} Gemini rows. {after} rows remain (GPT-4o + Llama).")