"""
Sort all_results.csv according to specified order:
1) task: lump_binary, size_multiclass, position_multiclass, multitask_all
2) sensor_config: all, tips_middle, tips
3) num_seconds: 7 to 1 (descending)
4) fold: 0 to 4
5) doctor_trials: 0 to 15
"""

from pathlib import Path
import pandas as pd


TASK_ORDER = ["lump_binary", "size_multiclass",
              "position_multiclass", "multitask_all"]
SENSOR_ORDER = ["all", "tips_middle", "tips"]
CV_ORDER = ["group", "plain"]
EVAL_ORDER = ["val", "doctors_test"]


def main():
    results_csv = Path("results/inceptiontime/all_results.csv")

    if not results_csv.exists():
        print(f"Error: {results_csv} not found.")
        return

    print(f"Loading {results_csv}...")
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} rows.")

    # Create categorical columns with specified order
    df["task_order"] = pd.Categorical(
        df["task"], categories=TASK_ORDER, ordered=True)
    df["sensor_order"] = pd.Categorical(
        df["sensor_config"], categories=SENSOR_ORDER, ordered=True)
    df["cv_order"] = pd.Categorical(
        df["cv_type"], categories=CV_ORDER, ordered=True)
    df["eval_order"] = pd.Categorical(
        df["eval_set"], categories=EVAL_ORDER, ordered=True)

    # Sort by specified columns
    df_sorted = df.sort_values(
        by=[
            "task_order",       # 1) task order
            "sensor_order",     # 2) sensor config order
            "cv_order",         # cv_type order (group before plain)
            "num_seconds",      # 3) num_seconds descending (7 to 1)
            "eval_order",       # eval_set order (val before doctors_test)
            "fold",             # 4) fold 0 to 4
            "doctor_trials",    # 5) doctor_trials 0 to 15
        ],
        ascending=[True, True, True, False, True,
                   True, True]  # num_seconds is descending
    )

    # Drop helper columns
    df_sorted = df_sorted.drop(
        columns=["task_order", "sensor_order", "cv_order", "eval_order"])

    # Save back
    df_sorted.to_csv(results_csv, index=False)
    print(f"Sorted and saved {len(df_sorted)} rows to {results_csv}")


if __name__ == "__main__":
    main()
