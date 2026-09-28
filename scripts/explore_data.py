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

    text_col = "turns"

    def is_empty(t):
        if t is None:
            return True
        try:
            return len(t) == 0 or all(x is None or str(x).strip() == "" for x in t)
        except TypeError:
            return str(t).strip() == ""

    empty = df[text_col].apply(is_empty)
    print(f"\nRows with empty '{text_col}': {empty.sum()} / {len(df)}")
    if cat_col:
        print("Empty rows per category:")
        print(df[empty][cat_col].value_counts())

    print("\nTurns per prompt:")
    print(df[text_col].apply(len).value_counts().sort_index())

    print("\nFirst row:")
    pd.set_option("display.max_colwidth", 300)
    print(df.iloc[0])

if __name__ == "__main__":
    main()
