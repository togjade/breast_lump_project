"""
plot_sensor_config.py
---------------------
Reproduce the four "sensor config" scatter plots from the paper using
all_results.csv produced by main.py.

Plots generated:
  1. Stratified Group K-fold CV – MTL
  2. Stratified K-fold CV       – MTL
  3. Stratified Group K-fold CV – STL
  4. Stratified K-fold CV       – STL

Each plot:
  x-axis    : recording duration (num_seconds)
  y-axis    : macro F1-score (%)  — averaged across folds
  colour    : task  (Lump Presence / Size / Location)
  facet_col : sensor configuration (Distal / Distal+Intermediate / All)
  facet_row : doctor trials added  (0 = val set only, no extra doctor data)
"""

import os
from pathlib import Path

import pandas as pd
import plotly.express as px

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / "results" / "inceptiontime"
CSV_PATH    = RESULTS_DIR / "all_results.csv"
OUTPUT_DIR  = RESULTS_DIR / "figures"

SAVE_FIGS = True   # set False to only call fig.show() without saving

# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------
TASK_MAP = {
    "lump_binary":        "Lump Presence",
    "size_multiclass":    "Size",
    "position_multiclass":"Location",
}

# Keys must match the sensor_config values stored in the CSV
SENSOR_CONFIG_MAP = {
    "distal":              "Distal",
    "distal_intermediate": "Distal and Intermediate",
    "all":                 "All",
}

SYMBOL_MAP = {
    "Lump Presence": "circle",
    "Size":          "square",
    "Location":      "star",
}

# Fixed column/row order for plotly category_orders
SENSOR_ORDER = ["distal", "distal_intermediate", "all"]
TASK_ORDER   = ["Lump Presence", "Size", "Location"]

# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------
GROUP_COLS = ["model", "task", "cv_type", "eval_set",
              "sensor_config", "num_seconds", "doctor_trials"]


def _base_filter(df: pd.DataFrame, cv_type: str) -> pd.DataFrame:
    """Keep only val-set rows with doctor_trials == 0 for the given cv_type."""
    return df[
        (df["cv_type"]    == cv_type) &
        (df["eval_set"]   == "val")   &
        (df["doctor_trials"] == 0)
    ].copy()


def preprocess_stl(df: pd.DataFrame, cv_type: str) -> pd.DataFrame:
    """
    Single-task learning subset.
    Uses the 'f1' column (macro F1 per fold) and averages across folds.
    """
    sub = _base_filter(df, cv_type)
    sub = sub[sub["task"].isin(TASK_MAP.keys())]

    agg = (sub.groupby(GROUP_COLS, as_index=False)["f1"]
               .mean())
    agg["f1"]  = agg["f1"] * 100          # proportion → percentage
    agg["task"] = agg["task"].map(TASK_MAP)
    return agg.reset_index(drop=True)


def preprocess_mtl(df: pd.DataFrame, cv_type: str) -> pd.DataFrame:
    """
    Multi-task learning subset.
    Reshapes the per-head columns (Lump_f1, Size_f1, Position_f1) into
    long format so the same plot_config() function can be reused.
    """
    sub = _base_filter(df, cv_type)
    sub = sub[sub["task"] == "multitask_all"]

    head_cols = {
        "Lump_f1":     "Lump Presence",
        "Size_f1":     "Size",
        "Position_f1": "Location",
    }

    frames = []
    for col, task_name in head_cols.items():
        part = sub[GROUP_COLS + [col]].copy()
        part = (part.groupby(GROUP_COLS, as_index=False)[col]
                    .mean()
                    .rename(columns={col: "f1"}))
        part["f1"]   = part["f1"] * 100   # proportion → percentage
        part["task"] = task_name
        frames.append(part)

    return pd.concat(frames, axis=0).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plot function
# ---------------------------------------------------------------------------
def plot_config(
    df:         pd.DataFrame,
    filename:   str,
    row_label:  str,
    save:       bool = SAVE_FIGS,
    output_dir: Path = OUTPUT_DIR,
) -> "go.Figure":
    """
    Scatter plot of F1-score vs. recording duration, faceted by sensor
    configuration (columns) and doctor_trials (rows).

    Parameters
    ----------
    df          : preprocessed dataframe with columns
                  [num_seconds, f1, task, sensor_config, doctor_trials]
    filename    : output filename (e.g. "sensor_config_val_group_MTL.pdf")
    row_label   : text to show on the facet-row annotation
                  (e.g. "Stratified Group <br>K-fold CV: MTL")
    save        : whether to write the figure to disk
    output_dir  : directory for saved figures
    """
    fig = px.scatter(
        df,
        x="num_seconds",
        y="f1",
        color="task",
        facet_col="sensor_config",
        facet_row="doctor_trials",
        facet_col_spacing=0.06,
        symbol="task",
        symbol_map=SYMBOL_MAP,
        category_orders={
            "sensor_config": SENSOR_ORDER,
            "task":          TASK_ORDER,
        },
        labels={
            "num_seconds": "Time [s]",
            "f1":          "F1-score (%)",
            "task":        "",
        },
    )

    # ---- clean up raw "col=value" / "row=value" annotation text ----------
    fig.for_each_annotation(
        lambda a: a.update(y=a.y + 0.04, text=a.text.split("=")[-1])
    )

    # ---- relabel sensor_config column headers ----------------------------
    for ann in fig.layout.annotations:
        txt = ann.text.strip()
        if txt in SENSOR_CONFIG_MAP:
            ann.text = SENSOR_CONFIG_MAP[txt]

    # ---- relabel doctor_trials row header --------------------------------
    # After the split("=")[-1] step the row annotation text is e.g. "0".
    # Replace it with the descriptive cv/learning-type label.
    for ann in fig.layout.annotations:
        txt = ann.text.strip()
        try:
            int(txt)   # the row annotation is a plain number
            ann.text = row_label
        except ValueError:
            pass

    # ---- axes & layout ---------------------------------------------------
    fig.update_yaxes(matches="y", showticklabels=True)

    fig.update_layout(
        font=dict(size=25),
        legend_title_text="",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.25,
            xanchor="center",
            x=0.6,
        ),
        margin=dict(l=0, r=45, t=50, b=0),
        width=1000,
        height=300,
        yaxis_range=[0, 100],
    )

    fig.update_traces(
        marker_size=12,
        selector=dict(mode="markers"),
        marker=dict(opacity=0.7),
        marker_line=dict(width=2),
    )

    fig.show()

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / filename
        # Expand bottom margin only for the saved file so the x-axis
        # "Time [s]" title is not clipped by kaleido (show() already fired)
        fig.write_image(str(out_path), height=300, width=1000)
        print(f"Saved: {out_path}")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Loading {CSV_PATH} …")
    df_all = pd.read_csv(CSV_PATH)

    # ---- build the four datasets -----------------------------------------
    df_group_MTL = preprocess_mtl(df_all, "group")
    df_plain_MTL = preprocess_mtl(df_all, "plain")
    df_group_STL = preprocess_stl(df_all, "group")
    df_plain_STL = preprocess_stl(df_all, "plain")

    # ---- generate the four paper plots -----------------------------------
    plot_config(
        df_group_MTL,
        "sensor_config_val_group_MTL.pdf",
        "Stratified Group <br>K-fold CV: MTL",
    )
    plot_config(
        df_plain_MTL,
        "sensor_config_val_plain_MTL.pdf",
        "Stratified <br>K-fold CV: MTL",
    )
    plot_config(
        df_group_STL,
        "sensor_config_val_group_STL.pdf",
        "Stratified Group <br>K-fold CV: STL",
    )
    plot_config(
        df_plain_STL,
        "sensor_config_val_plain_STL.pdf",
        "Stratified <br>K-fold CV: STL",
    )
