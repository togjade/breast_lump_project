"""Check class balance across all labels in both datasets."""

from pathlib import Path
import pandas as pd
import numpy as np


def prepare_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_pickle(path).reset_index(drop=True)
    df["Trial ID"] = df["Person No"].astype(str) + "_" + df["Trial No"].astype(str)
    df["Lump"] = df["Type"].apply(lambda x: 1 if x < 9 else 0)
    df["Size"] = df["Type"].apply(
        lambda x: 0 if x in [0, 1, 2] else 1 if x in [3, 4, 5] else 2 if x in [6, 7, 8] else 3
    )
    df["Position"] = df["Type"].apply(
        lambda x: 0 if x in [0, 3, 6] else 1 if x in [1, 4, 7] else 2 if x in [2, 5, 8] else 3
    )
    return df


def print_balance(df: pd.DataFrame, name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Dataset: {name}  —  {len(df)} total samples")
    print(f"{'='*60}")

    label_maps = {
        "Lump": {0: "No Lump", 1: "Lump"},
        "Size": {0: "Small", 1: "Medium", 2: "Big", 3: "No Lump"},
        "Position": {0: "Top", 1: "Middle", 2: "Bottom", 3: "No Lump"},
        "Type": {i: str(i) for i in range(13)},
    }

    for col, mapping in label_maps.items():
        counts = df[col].value_counts().sort_index()
        total = counts.sum()
        print(f"\n  {col}")
        print(f"  {'-'*40}")
        for val, cnt in counts.items():
            label = mapping.get(val, str(val))
            pct = cnt / total * 100
            bar = "█" * int(pct / 2)
            print(f"    {label:<12} {cnt:>5}  ({pct:5.1f}%)  {bar}")

        # Imbalance ratio (majority / minority)
        ratio = counts.max() / counts.min()
        print(f"    Imbalance ratio (max/min): {ratio:.2f}")


def main():
    data_path = Path("togzhan_data_labeled.pkl")
    doctors_path = Path("doctors_data_labeled.pkl")

    for path, name in [(data_path, "Togzhan (non-experts)"), (doctors_path, "Doctors (experts)")]:
        if not path.exists():
            print(f"  [SKIP] {path} not found")
            continue
        df = prepare_dataframe(str(path))
        print_balance(df, name)

    # Combined
    if data_path.exists() and doctors_path.exists():
        df1 = prepare_dataframe(str(data_path))
        df2 = prepare_dataframe(str(doctors_path))
        combined = pd.concat([df1, df2], ignore_index=True)
        print_balance(combined, "Combined (non-experts + doctors)")


if __name__ == "__main__":
    main()
