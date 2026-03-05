"""
Benchmark training speed across different DataLoader / training configurations.

Tests combinations of: batch_size, num_workers, pin_memory, AMP (mixed precision).
Reports epochs/sec and total time for a fixed number of training epochs.

Usage:
    python benchmark_speed.py
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from main import (
    DEVICE,
    SensorDataset,
    build_model,
    get_loss_functions,
    prepare_dataframe,
    set_seed,
    split_sensors,
)

# ── Configuration ──────────────────────────────────────────────────────────
DATA_PATH = "togzhan_data_labeled.pkl"
N_EPOCHS = 10  # enough to get a stable measurement
WARMUP_EPOCHS = 2  # excluded from timing (GPU warm-up, JIT compilation)

# Hyperparameter grid to sweep
BATCH_SIZES = [64, 128, 256, 512, 1024]
NUM_WORKERS_LIST = [0, 2, 4]
PIN_MEMORY_LIST = [True, False]
AMP_LIST = [True, False]

MODEL_NAME = "inceptiontime"
IN_CHANNELS = 15
NUM_SECONDS = 7


def build_loader(
    df: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    X = split_sensors(df["Data"].values, sensor_indices=None, num_seconds=NUM_SECONDS)
    features = torch.tensor(X, dtype=torch.float32)
    targets = {"Lump": torch.tensor(df["Lump"].values, dtype=torch.long)}
    ds = SensorDataset(features, targets)
    persistent = num_workers > 0
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent if num_workers > 0 else False,
    )


def run_training_loop(
    loader: DataLoader,
    use_amp: bool,
    n_epochs: int,
    warmup: int,
) -> Dict[str, float]:
    """Train for n_epochs, return timing stats (excluding warmup)."""
    set_seed(1337)
    model = build_model(MODEL_NAME, "binary", in_channels=IN_CHANNELS, num_classes=2)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = get_loss_functions("binary")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and torch.cuda.is_available())

    # Warmup (not timed)
    model.train()
    for _ in tqdm(range(warmup), desc="  warmup", leave=False):
        for batch in loader:
            inputs, targets = batch
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.float().to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(inputs).squeeze(), targets)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in tqdm(range(n_epochs), desc="  training", leave=False):
        for batch in loader:
            inputs, targets = batch
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.float().to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(inputs).squeeze(), targets)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "total_sec": round(elapsed, 4),
        "epochs_per_sec": round(n_epochs / elapsed, 2),
        "ms_per_epoch": round(elapsed / n_epochs * 1000, 2),
    }


def main():
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Training {N_EPOCHS} epochs (+ {WARMUP_EPOCHS} warmup) per config")
    print(f"Model: {MODEL_NAME} | Channels: {IN_CHANNELS} | Duration: {NUM_SECONDS}s")
    print()

    df = prepare_dataframe(DATA_PATH)
    print(f"Dataset: {len(df)} samples, input shape per sample: ({IN_CHANNELS}, {NUM_SECONDS * 160})")
    print()

    results: List[Dict] = []
    configs = list(itertools.product(BATCH_SIZES, NUM_WORKERS_LIST, PIN_MEMORY_LIST, AMP_LIST))
    total = len(configs)

    pbar = tqdm(configs, desc="Benchmarking", unit="config")
    for bs, nw, pm, amp in pbar:
        label = f"bs={bs} w={nw} pin={pm} amp={amp}"
        pbar.set_postfix_str(label)

        try:
            loader = build_loader(df, batch_size=bs, num_workers=nw, pin_memory=pm)
            stats = run_training_loop(loader, use_amp=amp, n_epochs=N_EPOCHS, warmup=WARMUP_EPOCHS)
            results.append({
                "batch_size": bs,
                "num_workers": nw,
                "pin_memory": pm,
                "amp": amp,
                **stats,
            })
            pbar.set_postfix_str(f"{label} | {stats['epochs_per_sec']:.0f} ep/s")
        except Exception as e:
            pbar.set_postfix_str(f"{label} | FAILED")
            results.append({
                "batch_size": bs,
                "num_workers": nw,
                "pin_memory": pm,
                "amp": amp,
                "total_sec": None,
                "epochs_per_sec": None,
                "ms_per_epoch": None,
            })

    # Sort by speed and display results
    results_df = pd.DataFrame(results).dropna(subset=["epochs_per_sec"])
    results_df = results_df.sort_values("epochs_per_sec", ascending=False).reset_index(drop=True)
    results_df.index += 1
    results_df.index.name = "rank"

    print("\n" + "=" * 90)
    print("RESULTS (sorted by speed, fastest first)")
    print("=" * 90)
    print(results_df.to_string())

    out_path = Path("benchmark_speed_results.csv")
    results_df.to_csv(out_path, index=True)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
