"""Step 0: inspect what the SPEED-Bench qualitative split actually contains."""
from datasets import load_dataset, get_dataset_config_names
import pandas as pd

REPO = "nvidia/SPEED-Bench"

def main():
    configs = get_dataset_config_names(REPO)
    print("Available configs:", configs)

    config = "qualitative" if "qualitative" in configs else configs[0]
    print(f"\nLoading config: {config}")
    ds = load_dataset(REPO, config)
    print(ds)

    split = list(ds.keys())[0]
    df = ds[split].to_pandas()

    print("\nColumns:", list(df.columns))
    print("Total rows:", len(df))

    cat_col = next((c for c in df.columns if "categ" in c.lower()), None)
    if cat_col:
        print(f"\nRows per {cat_col}:")
        print(df[cat_col].value_counts())

    text_col = next((c for c in df.columns
                     if any(k in c.lower() for k in ["prompt", "turns", "text", "question"])), None)
    if text_col:
        empty = df[text_col].isna() | (df[text_col].astype(str).str.strip().isin(["", "[]", "None"]))
        print(f"\nRows with empty '{text_col}': {empty.sum()} / {len(df)}")
        if cat_col:
            print("Empty rows per category:")
            print(df[empty][cat_col].value_counts())

    print("\nFirst row:")
    pd.set_option("display.max_colwidth", 300)
    print(df.iloc[0])

if __name__ == "__main__":
    main()
