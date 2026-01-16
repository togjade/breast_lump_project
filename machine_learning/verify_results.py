"""
Verify integrity of all_results.csv.
Checks for completeness and correct ordering of:
1) Tasks: lump_binary, size_multiclass, position_multiclass
2) Sensor configs: all, tips_middle, tips
3) Folds: 0 to 4
"""

from pathlib import Path
import pandas as pd


EXPECTED_TASKS = ["lump_binary", "size_multiclass", "position_multiclass"]
EXPECTED_SENSOR_CONFIGS = ["all", "tips_middle", "tips"]
EXPECTED_FOLDS = [0, 1, 2, 3, 4]
EXPECTED_CV_TYPES = ["group", "plain"]
EXPECTED_DURATIONS = [1, 2, 3, 4, 5, 6, 7]


def check_completeness(df: pd.DataFrame) -> dict:
    """Check if all expected combinations are present."""
    missing = []

    for task in EXPECTED_TASKS:
        for sensor_config in EXPECTED_SENSOR_CONFIGS:
            for cv_type in EXPECTED_CV_TYPES:
                for duration in EXPECTED_DURATIONS:
                    for fold in EXPECTED_FOLDS:
                        mask = (
                            (df["task"] == task) &
                            (df["sensor_config"] == sensor_config) &
                            (df["cv_type"] == cv_type) &
                            (df["num_seconds"] == duration) &
                            (df["fold"] == fold) &
                            (df["eval_set"] == "val") &
                            (df["doctor_trials"] == 0)
                        )
                        if not df[mask].any().any():
                            missing.append({
                                "task": task,
                                "sensor_config": sensor_config,
                                "cv_type": cv_type,
                                "num_seconds": duration,
                                "fold": fold
                            })

    return missing


def check_ordering(df: pd.DataFrame, eval_set: str = "val", doctor_trials: int = 0) -> list:
    """Check if rows follow expected ordering within each cv_type group."""
    issues = []

    filtered = df[(df["eval_set"] == eval_set) & (
        df["doctor_trials"] == doctor_trials)].copy()
    filtered = filtered.reset_index(drop=True)

    if filtered.empty:
        issues.append(
            "No rows found with specified eval_set and doctor_trials")
        return issues

    # Create expected order mapping
    task_order = {t: i for i, t in enumerate(EXPECTED_TASKS)}
    sensor_order = {s: i for i, s in enumerate(EXPECTED_SENSOR_CONFIGS)}

    # Check ordering within each cv_type group
    for cv_type in EXPECTED_CV_TYPES:
        # Check num_seconds in descending order (7 -> 1)
        for duration in sorted(EXPECTED_DURATIONS, reverse=True):
            group = filtered[(filtered["cv_type"] == cv_type)
                             & (filtered["num_seconds"] == duration)]
            if group.empty:
                continue

            prev_task_idx = -1
            prev_sensor_idx = -1
            prev_fold = -1

            for idx, row in group.iterrows():
                task = row["task"]
                sensor = row["sensor_config"]
                fold = row["fold"]

                if task not in task_order or sensor not in sensor_order:
                    continue

                task_idx = task_order[task]
                sensor_idx = sensor_order[sensor]

                # Check task ordering
                if task_idx < prev_task_idx:
                    issues.append(
                        f"Row {idx} ({cv_type}/{duration}s): Task '{task}' after later task")

                # Within same task, check sensor ordering
                if task_idx == prev_task_idx and sensor_idx < prev_sensor_idx:
                    issues.append(
                        f"Row {idx} ({cv_type}/{duration}s): Sensor '{sensor}' after later config in '{task}'")

                # Within same task and sensor, check fold ordering
                if task_idx == prev_task_idx and sensor_idx == prev_sensor_idx and fold < prev_fold:
                    issues.append(
                        f"Row {idx} ({cv_type}/{duration}s): Fold {fold} after fold {prev_fold} in {task}/{sensor}")

                prev_task_idx = task_idx
                prev_sensor_idx = sensor_idx
                prev_fold = fold

    return issues


def print_summary(df: pd.DataFrame):
    """Print summary of what's in the CSV."""
    print("\n=== CSV Summary ===")
    print(f"Total rows: {len(df)}")
    print(f"\nTasks present: {df['task'].unique().tolist()}")
    print(f"Sensor configs present: {df['sensor_config'].unique().tolist()}")
    print(f"CV types present: {df['cv_type'].unique().tolist()}")
    print(f"Durations present: {sorted(df['num_seconds'].unique().tolist())}")
    print(f"Folds present: {sorted(df['fold'].unique().tolist())}")
    print(f"Eval sets present: {df['eval_set'].unique().tolist()}")
    print(
        f"Doctor trials values: {sorted(df['doctor_trials'].unique().tolist())}")


def main():
    results_csv = Path("results/inceptiontime/all_results.csv")

    if not results_csv.exists():
        print(f"Error: {results_csv} not found.")
        return

    print(f"Loading {results_csv}...")
    df = pd.read_csv(results_csv, on_bad_lines="skip")

    print_summary(df)

    # Check completeness for baseline (doctor_trials=0, eval_set=val)
    print("\n=== Checking Completeness (baseline: eval_set='val', doctor_trials=0) ===")
    missing = check_completeness(df)
    if missing:
        print(f"MISSING {len(missing)} combinations:")
        for m in missing[:20]:  # Show first 20
            print(f"  - {m}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    else:
        print("✓ All expected combinations are present!")

    # Check ordering
    print("\n=== Checking Ordering ===")
    order_issues = check_ordering(df)
    if order_issues:
        print(f"ORDERING ISSUES ({len(order_issues)}):")
        for issue in order_issues[:20]:
            print(f"  - {issue}")
        if len(order_issues) > 20:
            print(f"  ... and {len(order_issues) - 20} more")
    else:
        print("✓ Ordering looks correct!")

    # Final verdict
    print("\n=== Verdict ===")
    if not missing and not order_issues:
        print("✓ CSV integrity verified successfully!")
    else:
        print("✗ Issues found. See above for details.")


if __name__ == "__main__":
    main()
