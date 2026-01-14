"""
Extract doctors_test evaluation results to CSV.
Filters for eval_set == 'doctors_test' and doctor_trials == 0.
"""

from pathlib import Path
import pandas as pd


def main():
    results_csv = Path("results/inceptiontime/all_results.csv")
    output_csv = Path("results/inceptiontime/doctors_test_results.csv")

    if not results_csv.exists():
        print(f"Error: {results_csv} not found.")
        return

    df = pd.read_csv(results_csv)
    filtered = df[
        (df["eval_set"] == "doctors_test") &
        (df["doctor_trials"] == 0) &
        (df["num_seconds"] == 7) &
        (df["sensor_config"] == "all")
    ]

    if filtered.empty:
        print("No rows found with eval_set='doctors_test', doctor_trials=0, num_seconds=7, and sensor_config='all'.")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)
    print(f"Extracted {len(filtered)} rows to: {output_csv.absolute()}")


if __name__ == "__main__":
    main()
