"""
Generate LaTeX tables from all_results.csv for research paper.
Creates 3 tables:
- Distal Phalanges (distal)
- Distal and Intermediate Phalanges (distal_intermediate)
- All (all)

Each table shows F1-scores (%) for:
- Single-task: Lump Presence, Size, Location
- Multitask: Lump Presence, Size, Location
For both Stratified Group K-Fold and Stratified K-Fold CV.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and prepare data from CSV (num_seconds=7 only)."""
    df = pd.read_csv(csv_path)
    df = df[df["num_seconds"] == 7]
    return df


def load_data_val_only(csv_path: Path) -> pd.DataFrame:
    """Load data filtered to val eval_set only (for fine-tuning tables)."""
    df = pd.read_csv(csv_path)
    df = df[(df["num_seconds"] == 7) & (df["eval_set"] == "val")]
    return df


def get_mean_f1(df: pd.DataFrame, task: str, cv_type: str, sensor_config: str, doctor_trials: int) -> float:
    """Get mean F1 score across folds for a specific configuration."""
    mask = (
        (df["task"] == task) &
        (df["cv_type"] == cv_type) &
        (df["sensor_config"] == sensor_config) &
        (df["doctor_trials"] == doctor_trials)
    )
    subset = df[mask]

    if subset.empty:
        return np.nan

    # For single-task, use 'f1' column
    # For multitask, use the appropriate head column
    if task == "lump_binary":
        return subset["f1"].mean() * 100
    elif task == "size_multiclass":
        return subset["f1"].mean() * 100
    elif task == "position_multiclass":
        return subset["f1"].mean() * 100
    elif task == "multitask_all":
        return None  # Will be handled separately
    return np.nan


def get_multitask_f1(df: pd.DataFrame, head: str, cv_type: str, sensor_config: str, doctor_trials: int) -> float:
    """Get mean F1 score for a specific multitask head."""
    mask = (
        (df["task"] == "multitask_all") &
        (df["cv_type"] == cv_type) &
        (df["sensor_config"] == sensor_config) &
        (df["doctor_trials"] == doctor_trials)
    )
    subset = df[mask]

    if subset.empty:
        return np.nan

    col_map = {"Lump": "Lump_f1", "Size": "Size_f1", "Position": "Position_f1"}
    col = col_map.get(head)
    if col and col in subset.columns:
        return subset[col].mean() * 100
    return np.nan


def format_value(val: float, is_best: bool = False) -> str:
    """Format a value for LaTeX output."""
    if pd.isna(val):
        return "-"
    formatted = f"{val:.1f}"
    if is_best:
        return f"\\textbf{{{formatted}}}"
    return formatted


def find_best_indices(values: list) -> list:
    """Find indices of maximum values (excluding NaN)."""
    valid = [(i, v) for i, v in enumerate(values) if not pd.isna(v)]
    if not valid:
        return []
    max_val = max(v for _, v in valid)
    return [i for i, v in valid if v == max_val]


def generate_initial_table(df: pd.DataFrame) -> str:
    """Generate the initial overview table comparing single-task vs multitask
    for val and doctors_test, with group and plain CV (doctor_trials=0, sensor_config=all)."""

    def _mean_f1(task: str, cv_type: str, eval_set: str, col: str = "f1") -> float:
        mask = (
            (df["task"] == task) &
            (df["cv_type"] == cv_type) &
            (df["eval_set"] == eval_set) &
            (df["sensor_config"] == "all") &
            (df["doctor_trials"] == 0)
        )
        subset = df[mask]
        if subset.empty or col not in subset.columns:
            return np.nan
        return subset[col].mean() * 100

    def _fmt(val: float) -> str:
        if pd.isna(val):
            return "-"
        return f"{val:.1f}"

    # --- Collect values: (eval_set, cv_type) ---
    configs = [
        ("val", "group"),
        ("val", "plain"),
        ("doctors_test", "group"),
        ("doctors_test", "plain"),
    ]

    data = {}
    for eval_set, cv_type in configs:
        key = (eval_set, cv_type)
        data[key] = {
            "st_lump": _mean_f1("lump_binary", cv_type, eval_set),
            "st_size": _mean_f1("size_multiclass", cv_type, eval_set),
            "st_pos": _mean_f1("position_multiclass", cv_type, eval_set),
            "mt_lump": _mean_f1("multitask_all", cv_type, eval_set, "Lump_f1"),
            "mt_size": _mean_f1("multitask_all", cv_type, eval_set, "Size_f1"),
            "mt_pos": _mean_f1("multitask_all", cv_type, eval_set, "Position_f1"),
        }

    def _vals(key):
        d = data[key]
        return (d["st_lump"], d["st_size"], d["st_pos"],
                d["mt_lump"], d["mt_size"], d["mt_pos"])

    vg = _vals(("val", "group"))
    vp = _vals(("val", "plain"))
    dg = _vals(("doctors_test", "group"))
    dp = _vals(("doctors_test", "plain"))

    def _mr(vals):
        """Format 6 values as multirow cells."""
        return " & ".join(f"\\multirow{{3}}{{*}}{{{_fmt(v)}}}" for v in vals)

    table = f"""\\begin{{table}}[!b]
\\centering
\\caption{{Lump Detection accuracy for single-task/multitask model with stratified group k-fold and stratified k-fold cross-validation. The table presents the averaged F1-score~(\\%) across five folds.}}
    \\resizebox{{1\\linewidth}}{{!}}{{
        \\begin{{tabular}}{{c c c c c| c c c}}
            \\toprule
            \\multirow{{3}}{{*}}{{Dataset}} & \\multirow{{3}}{{*}}{{{{Data Split }}}} &\\multicolumn{{3}}{{c}}{{Single-task}}& \\multicolumn{{3}}{{c}}{{{{Multitask}}}} \\\\ \\cmidrule(lr){{3-5}} \\cmidrule(lr){{6-8}} 
            &   &    {{Lump}}  & {{Size}} & {{Location}} & {{Lump }}     & {{Size}} & {{Location}} \\\\ 
            %------
            &   &     Presence & & & Presence   & &  \\\\ \\midrule
            %------
            & {{{{Stratified}}}}  & {_mr(vg)}\\\\ 
            %------
            &  {{{{Group}}}} & & & & & & \\\\
            %------
            Validation & {{{{k-fold}}}} & & & & & & \\\\
            %------
            Set & \\multirow{{3}}{{*}}{{{{{{Stratified}}}}}} & {_mr(vp)} \\\\
            & & & & & & & \\\\
            & {{{{k-fold}}}} & & & & & & \\\\ \\midrule
            %------------------------------
            & Stratified & {_mr(dg)}\\\\ 
            %------
            & Group & & & & & & \\\\  
            %------
            Clinician & {{{{k-fold}}}} & & & & & & \\\\
            %------
            Dataset& \\multirow{{3}}{{*}}{{{{{{Stratified}}}}}} & {_mr(dp)} \\\\
            %------
            & & & & & & & \\\\
            & {{{{k-fold}}}} & & & & & & \\\\
            \\bottomrule
        \\end{{tabular}}
    }}
\\label{{tab:initial}}
\\end{{table}}"""

    return table


def generate_table(df: pd.DataFrame, sensor_config: str, sensor_name: str) -> str:
    """Generate LaTeX table for a specific sensor configuration."""

    # Column structure:
    # Group CV: Single-task (Lump, Size, Position), Multitask (Lump, Size, Position)
    # Plain CV: Single-task (Lump, Size, Position), Multitask (Lump, Size, Position)

    doctor_trials_range = range(16)

    # Collect all values first to find the best ones
    all_values = {col_idx: [] for col_idx in range(12)}

    for dt in doctor_trials_range:
        row_vals = []
        # Group CV - Single-task
        row_vals.append(get_mean_f1(df, "lump_binary",
                        "group", sensor_config, dt))
        row_vals.append(get_mean_f1(df, "size_multiclass",
                        "group", sensor_config, dt))
        row_vals.append(get_mean_f1(df, "position_multiclass",
                        "group", sensor_config, dt))
        # Group CV - Multitask
        row_vals.append(get_multitask_f1(
            df, "Lump", "group", sensor_config, dt))
        row_vals.append(get_multitask_f1(
            df, "Size", "group", sensor_config, dt))
        row_vals.append(get_multitask_f1(
            df, "Position", "group", sensor_config, dt))
        # Plain CV - Single-task
        row_vals.append(get_mean_f1(df, "lump_binary",
                        "plain", sensor_config, dt))
        row_vals.append(get_mean_f1(df, "size_multiclass",
                        "plain", sensor_config, dt))
        row_vals.append(get_mean_f1(df, "position_multiclass",
                        "plain", sensor_config, dt))
        # Plain CV - Multitask
        row_vals.append(get_multitask_f1(
            df, "Lump", "plain", sensor_config, dt))
        row_vals.append(get_multitask_f1(
            df, "Size", "plain", sensor_config, dt))
        row_vals.append(get_multitask_f1(
            df, "Position", "plain", sensor_config, dt))

        for col_idx, val in enumerate(row_vals):
            all_values[col_idx].append(val)

    # Find best values for each column
    best_indices = {col_idx: find_best_indices(
        vals) for col_idx, vals in all_values.items()}

    # Generate table rows
    rows = []
    for row_idx, dt in enumerate(doctor_trials_range):
        row_data = []
        for col_idx in range(12):
            val = all_values[col_idx][row_idx]
            is_best = row_idx in best_indices[col_idx]
            row_data.append(format_value(val, is_best))

        row_str = f"            {dt} & " + " & ".join(row_data) + " \\\\"
        rows.append(row_str)

    # Build the full table
    table = f"""\\begin{{table*}}[t]
    \\centering
    \\caption{{Lump Detection F1-scores~(\\%) for Single-task/Multitask Model with Stratified Group K-Fold Cross-Validation and Stratified K-Fold Cross-Validation Data Splits and Fine-Tuning ({sensor_name}).}}
    \\resizebox{{.9\\textwidth}}{{!}}{{
    \\begin{{tabular}}{{c c c c| c c c| c c c| c c c}}
        \\toprule
        \\multirow{{2}}{{*}}{{Added}} & \\multicolumn{{6}}{{c}}{{Stratified Group K-Fold Cross-Validation}} & \\multicolumn{{6}}{{c}}{{Stratified K-Fold Cross-Validation}}
        \\\\ \\cmidrule(lr){{3-5}}  \\cmidrule(lr){{9-11}}
        Trials &\\multicolumn{{3}}{{c}}{{Single-task}} &\\multicolumn{{3}}{{c}}{{Multitask}} &\\multicolumn{{3}}{{c}}{{Single-task}} &\\multicolumn{{3}}{{c}}{{Multitask}}
        \\\\  \\cmidrule(lr){{2-13}} 
        & {{Lump Presence}} & {{Size}} & {{Location}} & {{Lump Presence}} & {{Size}} & {{Location}} & {{Lump Presence}} & {{Size}} & {{Location}} & {{Lump Presence}} & {{Size}} & {{Location}}\\\\ 
         \\midrule
{chr(10).join(rows)}
                                  
        \\bottomrule
    \\end{{tabular}}
    }}
    \\label{{tab:fine_tuning_{sensor_config}}}
\\end{{table*}}"""

    return table


def main():
    # Try both possible paths
    csv_path = Path("results/inceptiontime/all_results.csv")
    if not csv_path.exists():
        csv_path = Path("all_results.csv")

    if not csv_path.exists():
        print(f"Error: CSV file not found.")
        return

    print(f"Loading {csv_path}...")
    df_all = load_data(csv_path)
    df_val = df_all[df_all["eval_set"] == "val"]
    print(f"Total rows (7s): {len(df_all)}, val-only: {len(df_val)}")

    output_dir = Path("results/inceptiontime")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Initial overview table ---
    print("Generating initial overview table...")
    initial_table = generate_initial_table(df_all)
    print("  Done.")

    # --- Fine-tuning tables per sensor config ---

    sensor_configs = [
        ("distal", "Distal Phalanges"),
        ("distal_intermediate", "Distal and Intermediate Phalanges"),
        ("all", "All Sensors"),
    ]

    all_tables = [initial_table]
    for sensor_config, sensor_name in sensor_configs:
        print(f"Generating table for {sensor_name} ({sensor_config})...")
        table = generate_table(df_val, sensor_config, sensor_name)
        all_tables.append(table)
        print(f"  Done.")

    # Save all tables to a single file
    output_path = output_dir / "latex_tables.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("% LaTeX tables generated from all_results.csv\n")
        f.write("% Requires: booktabs, multirow packages\n\n")
        f.write("\n\n".join(all_tables))

    print(f"\nSaved all tables to: {output_path}")


if __name__ == "__main__":
    main()
