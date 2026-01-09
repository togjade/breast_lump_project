from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)


set_seed(1337)

if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} GPU)")
else:
    print("CUDA not available, using CPU.")


def prepare_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_pickle(path).reset_index(drop=True)
    df["Trial ID"] = df["Person No"].astype(str) + "_" + df["Trial No"].astype(str)
    df["Lump"] = df["Type"].apply(lambda x: 1 if x < 9 else 0)
    df["Size"] = df["Type"].apply(lambda x: 0 if x in [0, 1, 2] else 1 if x in [3, 4, 5] else 2 if x in [6, 7, 8] else 3)
    df["Position"] = df["Type"].apply(lambda x: 0 if x in [0, 3, 6] else 1 if x in [1, 4, 7] else 2 if x in [2, 5, 8] else 3)
    return df


def split_sensors(data: Iterable[np.ndarray], sensor_indices: Optional[List[int]] = None, num_seconds: int = 7) -> np.ndarray:
    sensor_indices = sensor_indices if sensor_indices is not None else list(range(15))
    arr = np.asarray(list(data), dtype=np.float32)
    sample_len = arr.shape[1] // 15
    arr = arr.reshape(-1, 15, sample_len)
    max_len = min(num_seconds * 160, arr.shape[2])
    selected = arr[:, sensor_indices, :max_len]
    return selected


class SensorDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: Dict[str, torch.Tensor]):
        self.features = features
        self.targets = targets
        self.multi_task = len(targets) > 1
        self.keys = list(targets.keys())

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        x = self.features[idx]
        if self.multi_task:
            return x, tuple(self.targets[k][idx] for k in self.keys)
        return x, self.targets[self.keys[0]][idx]


def build_dataloader(df: pd.DataFrame, label_cols: List[str], batch_size: int, sensor_indices: Optional[List[int]] = None, num_seconds: int = 7, shuffle: bool = False) -> DataLoader:
    X = split_sensors(df["Data"].values, sensor_indices=sensor_indices, num_seconds=num_seconds)
    features = torch.tensor(X, dtype=torch.float32)
    targets = {col: torch.tensor(df[col].values, dtype=torch.long) for col in label_cols}
    dataset = SensorDataset(features, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {"f1": f1, "precision": precision, "recall": recall, "accuracy": acc}


def save_confusion_matrix(labels: List[str], out_path: Path, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, cm_matrix: Optional[np.ndarray] = None) -> None:
    if cm_matrix is None:
        if y_true is None or y_pred is None:
            raise ValueError("Either cm_matrix or both y_true and y_pred must be provided.")
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    else:
        cm = cm_matrix
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# Unified columns to keep CSV schema consistent across tasks
ALL_RESULT_COLUMNS = [
    "model",
    "task",
    "cv_type",
    "eval_set",
    "sensor_config",
    "num_seconds",
    "fold",
    "doctor_trials",
    # Primary (single-task) metrics or combined view
    "f1",
    "precision",
    "recall",
    "accuracy",
    # Head 1 (Lump or primary)
    "Lump_f1",
    "Lump_precision",
    "Lump_recall",
    "Lump_acc",
    # Head 2 (Size)
    "Size_f1",
    "Size_precision",
    "Size_recall",
    "Size_acc",
    # Head 3 (Position)
    "Position_f1",
    "Position_precision",
    "Position_recall",
    "Position_acc",
    # Combined averages across heads
    "combined_f1",
    "combined_precision",
    "combined_recall",
    "combined_accuracy",
]


def normalize_row(row: Dict) -> Dict:
    """Project a heterogeneous result dict onto the unified schema."""
    norm = {k: None for k in ALL_RESULT_COLUMNS}
    for k, v in row.items():
        if k in norm:
            norm[k] = v
    return norm


def safe_int(val) -> Optional[int]:
    try:
        return int(val)
    except Exception:
        return None


def read_results_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        df = None
    if df is None:
        return None

    if set(df.columns).issubset(set(ALL_RESULT_COLUMNS)):
        # Add any missing columns from older files without treating the header row as data.
        for col in ALL_RESULT_COLUMNS:
            if col not in df.columns:
                df[col] = "val" if col == "eval_set" else None
        df["eval_set"] = df.get("eval_set", "val").fillna("val")
        return df[ALL_RESULT_COLUMNS]

    # Fallback: coerce with provided names if columns are misaligned.
    df = pd.read_csv(path, engine="python", on_bad_lines="skip", header=None, names=ALL_RESULT_COLUMNS)
    df["eval_set"] = df.get("eval_set", "val").fillna("val")
    return df[ALL_RESULT_COLUMNS]


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -math.inf
        self.counter = 0

    def step(self, value: float) -> bool:
        if value > self.best + self.min_delta:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter > self.patience


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: Tuple[int, int, int] = (10, 20, 40), bottleneck_channels: int = 32):
        super().__init__()
        self.use_bottleneck = in_channels > 1
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1) if self.use_bottleneck else nn.Identity()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                bottleneck_channels if self.use_bottleneck else in_channels,
                out_channels // 4,
                kernel_size=k,
                padding="same",
                bias=False,
            )
            for k in kernel_sizes
        ])
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_b = self.bottleneck(x)
        conv_outs = [conv(x_b) for conv in self.convs]
        pool = self.pool_conv(self.maxpool(x))
        x_cat = torch.cat(conv_outs + [pool], dim=1)
        return self.relu(self.bn(x_cat))


class InceptionTime(nn.Module):
    def __init__(self, in_channels: int, num_blocks: int = 3, out_channels: int = 32):
        super().__init__()
        blocks = []
        ch = in_channels
        for _ in range(num_blocks):
            block = InceptionBlock(ch, out_channels)
            blocks.append(nn.Sequential(block, InceptionBlock(out_channels, out_channels)))
            ch = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.shortcut = nn.ModuleList([nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1) for i in range(num_blocks)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(out_channels) for _ in range(num_blocks)])
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out_features = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            residual = self.bn[i](self.shortcut[i](x))
            x = block(x)
            x = self.relu(x + residual)
        x = self.gap(x).squeeze(-1)
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: Optional[int] = None):
        super().__init__()
        padding = padding if padding is not None else kernel_size // 2
        self.depth = nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch, bias=False)
        self.point = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth(x)
        x = self.point(x)
        return self.act(self.bn(x))


class XceptionTime(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        self.block1 = DepthwiseSeparableConv(in_channels, base_channels, kernel_size=7)
        self.block2 = DepthwiseSeparableConv(base_channels, base_channels, kernel_size=7)
        self.block3 = DepthwiseSeparableConv(base_channels, base_channels * 2, kernel_size=7, stride=2)
        self.block4 = DepthwiseSeparableConv(base_channels * 2, base_channels * 2, kernel_size=5)
        self.block5 = DepthwiseSeparableConv(base_channels * 2, base_channels * 4, kernel_size=3, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out_features = base_channels * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gap(x).squeeze(-1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.relu(out + identity)
        return out


class XResNet1d34(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(base_channels, base_channels, blocks=3)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, blocks=4, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, blocks=6, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out_features = base_channels * 4

    def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [ResidualBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x).squeeze(-1)
        return x


class mWDN(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 24, levels: int = 3):
        super().__init__()
        blocks = []
        ch = in_channels
        for i in range(levels):
            blocks.append(nn.Sequential(
                nn.Conv1d(ch, base_channels * (2 ** i), kernel_size=5, padding=2, groups=1, bias=False),
                nn.BatchNorm1d(base_channels * (2 ** i)),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ))
            ch = base_channels * (2 ** i)
        self.blocks = nn.ModuleList(blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out_features = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.gap(x).squeeze(-1)
        return x


class ResCNN(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        self.block1 = ResidualBlock(in_channels, base_channels)
        self.block2 = ResidualBlock(base_channels, base_channels * 2, stride=2)
        self.block3 = ResidualBlock(base_channels * 2, base_channels * 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out_features = base_channels * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).squeeze(-1)
        return x


class FCN(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels, base_channels, kernel_size=8, padding=4), nn.BatchNorm1d(base_channels), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, padding=2), nn.BatchNorm1d(base_channels * 2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1), nn.BatchNorm1d(base_channels * 2), nn.ReLU())
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out_features = base_channels * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x).squeeze(-1)
        return x


class LSTMBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, layers: int = 2, bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden, num_layers=layers, batch_first=True, bidirectional=bidirectional)
        self.out_features = hidden * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        return out


class LSTMFCN(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.fcn = FCN(in_channels, base_channels=32)
        self.lstm = LSTMBackbone(in_channels, hidden=hidden)
        self.out_features = self.fcn.out_features + self.lstm.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fcn_feat = self.fcn(x)
        lstm_feat = self.lstm(x)
        return torch.cat([fcn_feat, lstm_feat], dim=1)


def build_backbone(name: str, in_channels: int) -> nn.Module:
    name = name.lower()
    if name == "inceptiontime":
        return InceptionTime(in_channels=in_channels)
    if name == "xceptiontime":
        return XceptionTime(in_channels=in_channels)
    if name in {"xresnet1d34", "xresnet"}:
        return XResNet1d34(in_channels=in_channels)
    if name == "mwdn":
        return mWDN(in_channels=in_channels)
    if name == "rescnn":
        return ResCNN(in_channels=in_channels)
    if name == "resnet":
        return ResCNN(in_channels=in_channels)
    if name == "fcn":
        return FCN(in_channels=in_channels)
    if name == "lstm":
        return LSTMBackbone(in_channels=in_channels)
    if name == "lstm-fcn" or name == "lstm_fcn":
        return LSTMFCN(in_channels=in_channels)
    raise ValueError(f"Unknown model name: {name}")


class ClassifierHead(nn.Module):
    def __init__(self, in_features: int, task_type: Literal["binary", "multiclass"], num_classes: int = 2):
        super().__init__()
        out_dim = 1 if task_type == "binary" else num_classes
        self.fc = nn.Linear(in_features, out_dim)
        self.task_type = task_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MultiTaskHead(nn.Module):
    def __init__(self, in_features: int, num_classes: Dict[str, int]):
        super().__init__()
        self.lump = nn.Linear(in_features, 1)
        self.size = nn.Linear(in_features, num_classes["Size"])
        self.position = nn.Linear(in_features, num_classes["Position"])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"Lump": self.lump(x), "Size": self.size(x), "Position": self.position(x)}


class ModelWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, task_type: Literal["binary", "multiclass", "multitask"]):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.task_type = task_type

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        return self.head(feats)


def build_model(name: str, task_type: Literal["binary", "multiclass"], in_channels: int, num_classes: int) -> ModelWrapper:
    backbone = build_backbone(name, in_channels)
    head = ClassifierHead(backbone.out_features, task_type=task_type, num_classes=num_classes)
    return ModelWrapper(backbone, head, task_type)


def build_multitask_model(name: str, in_channels: int, num_classes: Dict[str, int]) -> ModelWrapper:
    backbone = build_backbone(name, in_channels)
    head = MultiTaskHead(backbone.out_features, num_classes)
    return ModelWrapper(backbone, head, task_type="multitask")


def get_loss_functions(task_type: Literal["binary", "multiclass", "multitask"], class_weights: Optional[Dict[str, torch.Tensor]] = None):
    if task_type == "binary":
        return nn.BCEWithLogitsLoss()
    if task_type == "multiclass":
        weight = None if class_weights is None else class_weights.get("target")
        return nn.CrossEntropyLoss(weight=weight)
    if task_type == "multitask":
        lump_w = None if class_weights is None else class_weights.get("Lump")
        size_w = None if class_weights is None else class_weights.get("Size")
        pos_w = None if class_weights is None else class_weights.get("Position")
        return {
            "Lump": nn.BCEWithLogitsLoss(weight=lump_w),
            "Size": nn.CrossEntropyLoss(weight=size_w),
            "Position": nn.CrossEntropyLoss(weight=pos_w),
        }
    raise ValueError(f"Unsupported task_type {task_type}")


def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, criterion, task_type: Literal["binary", "multiclass", "multitask"], desc: str = "train"):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc=desc, leave=False):
        optimizer.zero_grad()
        if task_type == "multitask":
            inputs, (lump, size, pos) = batch
            inputs = inputs.to(DEVICE)
            lump = lump.float().to(DEVICE)
            size = size.to(DEVICE)
            pos = pos.to(DEVICE)
            outputs = model(inputs)
            loss = criterion["Lump"](outputs["Lump"].squeeze(), lump)
            loss += criterion["Size"](outputs["Size"], size)
            loss += criterion["Position"](outputs["Position"], pos)
        else:
            inputs, targets = batch
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(inputs)
            if task_type == "binary":
                targets = targets.float()
                loss = criterion(outputs.squeeze(), targets)
            else:
                loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)


def predict_logits(model: nn.Module, dataloader: DataLoader, task_type: Literal["binary", "multiclass", "multitask"], desc: str = "eval"):
    model.eval()
    logits_list, targets_list = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            if task_type == "multitask":
                inputs, target_tuple = batch
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                logits_list.append({
                    "Lump": outputs["Lump"].cpu(),
                    "Size": outputs["Size"].cpu(),
                    "Position": outputs["Position"].cpu(),
                })
                targets_list.append({
                    "Lump": target_tuple[0],
                    "Size": target_tuple[1],
                    "Position": target_tuple[2],
                })
            else:
                inputs, targets = batch
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                logits_list.append(outputs.cpu())
                targets_list.append(targets)
    return logits_list, targets_list


def gather_predictions(logits_list, targets_list, task_type: Literal["binary", "multiclass", "multitask"]):
    if task_type == "multitask":
        preds, trues = {}, {}
        for key in ["Lump", "Size", "Position"]:
            pred_chunks = [torch.sigmoid(x[key]).squeeze() if key == "Lump" else torch.argmax(x[key], dim=1) for x in logits_list]
            true_chunks = [t[key] for t in targets_list]
            preds[key] = torch.cat(pred_chunks).numpy()
            if key == "Lump":
                preds[key] = (preds[key] > 0.5).astype(int)
            trues[key] = torch.cat(true_chunks).numpy()
        return preds, trues
    logits = torch.cat(logits_list)
    targets = torch.cat(targets_list).numpy()
    if task_type == "binary":
        preds = torch.sigmoid(logits.squeeze()).numpy()
        preds = (preds > 0.5).astype(int)
    else:
        preds = torch.argmax(logits, dim=1).numpy()
    return preds, targets


def evaluate(model: nn.Module, dataloader: DataLoader, task_type: Literal["binary", "multiclass", "multitask"], label_names: Dict[str, List[str]]):
    logits, targets = predict_logits(model, dataloader, task_type, desc="val")
    if task_type == "multitask":
        preds, trues = gather_predictions(logits, targets, task_type)
        metrics = {}
        cms = {}
        for key in ["Lump", "Size", "Position"]:
            avg = "binary" if key == "Lump" else "macro"
            metrics[key] = compute_metrics(trues[key], preds[key], average="macro" if avg == "macro" else "binary")
            cms[key] = confusion_matrix(trues[key], preds[key], labels=list(range(len(label_names[key]))))
        metrics["combined_f1"] = float(np.mean([metrics[k]["f1"] for k in ["Lump", "Size", "Position"]]))
        return metrics, cms
    preds, trues = gather_predictions(logits, targets, task_type)
    average = "binary" if task_type == "binary" else "macro"
    metrics = compute_metrics(trues, preds, average=average)
    cm = confusion_matrix(trues, preds, labels=list(range(len(next(iter(label_names.values()))))))
    return metrics, cm


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, task_type: Literal["binary", "multiclass", "multitask"], label_names: Dict[str, List[str]], lr: float = 1e-3, weight_decay: float = 1e-4, max_epochs: int = 100, patience: int = 15, fold_desc: str = ""):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = get_loss_functions(task_type)
    early = EarlyStopping(patience=patience)
    best_state = None
    best_score = -math.inf
    history = []

    for epoch in range(max_epochs):
        epoch_desc = f"{fold_desc}epoch {epoch+1}/{max_epochs}" if fold_desc else f"epoch {epoch+1}/{max_epochs}"
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, task_type, desc=f"train | {epoch_desc}")
        val_metrics, _ = evaluate(model, val_loader, task_type, label_names)
        score = val_metrics["combined_f1"] if task_type == "multitask" else val_metrics["f1"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_score": score})
        stop = early.step(score)
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if stop:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@dataclass
class TaskSpec:
    name: str
    label_cols: List[str]
    task_type: Literal["binary", "multiclass", "multitask"]
    num_classes: Dict[str, int]
    label_names: Dict[str, List[str]]


def stratified_splitter(cv_type: Literal["group", "plain"], n_splits: int = 5, seed: int = 1337):
    if cv_type == "group":
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def get_stratify_target(df: pd.DataFrame, task: TaskSpec) -> np.ndarray:
    if task.task_type == "multitask":
        return df["Lump"].values
    return df[task.label_cols[0]].values


def run_cv(
    df: pd.DataFrame,
    model_name: str,
    task: TaskSpec,
    cv_type: Literal["group", "plain"],
    sensor_config: str,
    n_splits: int,
    batch_size: int,
    results_dir: Path,
    doctor_trials: int = 0,
    doctors_df: Optional[pd.DataFrame] = None,
    num_seconds: int = 7,
    sensor_indices: Optional[List[int]] = None,
    skip_completed: Optional[set] = None,
    results_path: Optional[Path] = None,
    external_test_df: Optional[pd.DataFrame] = None,
    external_eval_name: str = "doctors_test",
):
    results = []
    df_use = df.copy()
    if doctor_trials > 0 and doctors_df is not None:
        extra = doctors_df.groupby("Trial ID").head(doctor_trials)
        df_use = pd.concat([df_use, extra], axis=0).reset_index(drop=True)

    groups = df_use["Person No"] if cv_type == "group" else None
    y_strat = get_stratify_target(df_use, task)
    splitter = stratified_splitter(cv_type, n_splits=n_splits)

    fold_iter = splitter.split(df_use, y_strat, groups)
    for fold, (train_idx, val_idx) in enumerate(tqdm(fold_iter, total=n_splits, desc=f"{task.name}-{model_name}-{cv_type}-folds")):
        key = (task.name, model_name, cv_type, sensor_config, num_seconds, fold, doctor_trials)
        if skip_completed and key in skip_completed:
            continue
        train_df = df_use.iloc[train_idx].reset_index(drop=True)
        val_df = df_use.iloc[val_idx].reset_index(drop=True)
        train_loader = build_dataloader(train_df, task.label_cols, batch_size=batch_size, sensor_indices=sensor_indices, num_seconds=num_seconds, shuffle=True)
        val_loader = build_dataloader(val_df, task.label_cols, batch_size=batch_size, sensor_indices=sensor_indices, num_seconds=num_seconds)

        if task.task_type == "multitask":
            model = build_multitask_model(model_name, in_channels=len(sensor_indices or list(range(15))), num_classes=task.num_classes)
        else:
            n_classes = list(task.num_classes.values())[0]
            model = build_model(model_name, task.task_type, in_channels=len(sensor_indices or list(range(15))), num_classes=n_classes)
        model.to(DEVICE)
        model, _ = fit(model, train_loader, val_loader, task.task_type, task.label_names, fold_desc=f"fold {fold+1}/{n_splits} | ")
        metrics, cms = evaluate(model, val_loader, task.task_type, task.label_names)

        fold_result = {
            "model": model_name,
            "task": task.name,
            "cv_type": cv_type,
            "eval_set": "val",
            "sensor_config": sensor_config,
            "num_seconds": num_seconds,
            "fold": fold,
            "doctor_trials": doctor_trials,
        }
        if task.task_type == "multitask":
            for key in ["Lump", "Size", "Position"]:
                fold_result.update({f"{key}_f1": metrics[key]["f1"], f"{key}_precision": metrics[key]["precision"], f"{key}_recall": metrics[key]["recall"], f"{key}_acc": metrics[key]["accuracy"]})
                cm_path = results_dir / f"cm_{task.name}_{cv_type}_{sensor_config}_{num_seconds}s_{model_name}_fold{fold}_{key}_val.png"
                save_confusion_matrix(task.label_names[key], cm_path, cm_matrix=cms[key])
            fold_result["combined_f1"] = metrics["combined_f1"]
            fold_result["combined_precision"] = float(np.mean([metrics[k]["precision"] for k in ["Lump", "Size", "Position"]]))
            fold_result["combined_recall"] = float(np.mean([metrics[k]["recall"] for k in ["Lump", "Size", "Position"]]))
            fold_result["combined_accuracy"] = float(np.mean([metrics[k]["accuracy"] for k in ["Lump", "Size", "Position"]]))
            # For clarity, set primary slots to combined metrics in multitask rows
            fold_result["f1"] = fold_result["combined_f1"]
            fold_result["precision"] = fold_result["combined_precision"]
            fold_result["recall"] = fold_result["combined_recall"]
            fold_result["accuracy"] = fold_result["combined_accuracy"]
        else:
            fold_result.update({"f1": metrics["f1"], "precision": metrics["precision"], "recall": metrics["recall"], "accuracy": metrics["accuracy"]})
            cm_path = results_dir / f"cm_{task.name}_{cv_type}_{sensor_config}_{num_seconds}s_{model_name}_fold{fold}_val.png"
            save_confusion_matrix(task.label_names[task.label_cols[0]], cm_path, cm_matrix=cms)
            fold_result["combined_f1"] = fold_result["f1"]
            fold_result["combined_precision"] = fold_result["precision"]
            fold_result["combined_recall"] = fold_result["recall"]
            fold_result["combined_accuracy"] = fold_result["accuracy"]
        fold_row = normalize_row(fold_result)
        results.append(fold_row)
        if results_path is not None:
            results_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([fold_row])[ALL_RESULT_COLUMNS].to_csv(
                results_path,
                mode="a",
                header=not results_path.exists(),
                index=False,
            )

        # Optional external test on doctors data (only when not already augmenting with doctors trials)
        if external_test_df is not None and doctor_trials == 0:
            test_loader = build_dataloader(external_test_df, task.label_cols, batch_size=batch_size, sensor_indices=sensor_indices, num_seconds=num_seconds)
            test_metrics, test_cms = evaluate(model, test_loader, task.task_type, task.label_names)
            test_result = {
                "model": model_name,
                "task": task.name,
                "cv_type": cv_type,
                "eval_set": external_eval_name,
                "sensor_config": sensor_config,
                "num_seconds": num_seconds,
                "fold": fold,
                "doctor_trials": doctor_trials,
            }
            if task.task_type == "multitask":
                for key in ["Lump", "Size", "Position"]:
                    test_result.update({f"{key}_f1": test_metrics[key]["f1"], f"{key}_precision": test_metrics[key]["precision"], f"{key}_recall": test_metrics[key]["recall"], f"{key}_acc": test_metrics[key]["accuracy"]})
                    cm_path = results_dir / f"cm_{task.name}_{cv_type}_{sensor_config}_{num_seconds}s_{model_name}_fold{fold}_{key}_{external_eval_name}.png"
                    save_confusion_matrix(task.label_names[key], cm_path, cm_matrix=test_cms[key])
                test_result["combined_f1"] = test_metrics["combined_f1"]
                test_result["combined_precision"] = float(np.mean([test_metrics[k]["precision"] for k in ["Lump", "Size", "Position"]]))
                test_result["combined_recall"] = float(np.mean([test_metrics[k]["recall"] for k in ["Lump", "Size", "Position"]]))
                test_result["combined_accuracy"] = float(np.mean([test_metrics[k]["accuracy"] for k in ["Lump", "Size", "Position"]]))
                test_result["f1"] = test_result["combined_f1"]
                test_result["precision"] = test_result["combined_precision"]
                test_result["recall"] = test_result["combined_recall"]
                test_result["accuracy"] = test_result["combined_accuracy"]
            else:
                test_result.update({"f1": test_metrics["f1"], "precision": test_metrics["precision"], "recall": test_metrics["recall"], "accuracy": test_metrics["accuracy"]})
                cm_path = results_dir / f"cm_{task.name}_{cv_type}_{sensor_config}_{num_seconds}s_{model_name}_fold{fold}_{external_eval_name}.png"
                save_confusion_matrix(task.label_names[task.label_cols[0]], cm_path, cm_matrix=test_cms)
                test_result["combined_f1"] = test_result["f1"]
                test_result["combined_precision"] = test_result["precision"]
                test_result["combined_recall"] = test_result["recall"]
                test_result["combined_accuracy"] = test_result["accuracy"]
            test_row = normalize_row(test_result)
            results.append(test_row)
            if results_path is not None:
                results_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([test_row])[ALL_RESULT_COLUMNS].to_csv(
                    results_path,
                    mode="a",
                    header=not results_path.exists(),
                    index=False,
                )
    return results


def run_benchmark(
    df: pd.DataFrame,
    doctors_df: pd.DataFrame,
    results_dir: Path,
    sensor_configs: Dict[str, List[int]],
    durations: List[int],
    batch_size: int = 32,
    n_splits: int = 5,
):
    binary_task = TaskSpec(
        name="binary_lump",
        label_cols=["Lump"],
        task_type="binary",
        num_classes={"Lump": 2},
        label_names={"Lump": ["No Lump", "Lump"]},
    )
    models = ["inceptiontime", "xceptiontime", "xresnet1d34", "mwdn", "rescnn", "lstm-fcn", "fcn", "lstm"]
    results_path = results_dir / "benchmark_results.csv"
    skip_completed = set()
    prev = read_results_csv(results_path)
    if prev is not None:
        for _, r in prev.iterrows():
            f = safe_int(r.get("fold"))
            d = safe_int(r.get("doctor_trials"))
            sec = safe_int(r.get("num_seconds"))
            sensor_conf = r.get("sensor_config") if pd.notna(r.get("sensor_config")) else "all"
            eval_set = r.get("eval_set") if pd.notna(r.get("eval_set")) else "val"
            if eval_set != "val":
                continue
            if f is None or d is None or sec is None:
                continue
            skip_completed.add((r.get("task"), r.get("model"), r.get("cv_type"), sensor_conf, sec, f, d))

    all_results = []
    for model_name in models:
        for sensor_conf, sensor_indices in sensor_configs.items():
            for duration in durations:
                for cv_type in ["group", "plain"]:
                    fold_results = run_cv(
                        df,
                        model_name,
                        binary_task,
                        cv_type=cv_type,
                        sensor_config=sensor_conf,
                        n_splits=n_splits,
                        batch_size=batch_size,
                        results_dir=results_dir,
                        doctors_df=doctors_df,
                        num_seconds=duration,
                        sensor_indices=sensor_indices,
                        skip_completed=skip_completed,
                        results_path=results_path,
                        external_test_df=doctors_df,
                    )
                    all_results.extend(fold_results)
    return all_results


def run_full_experiments(
    df: pd.DataFrame,
    doctors_df: pd.DataFrame,
    results_dir: Path,
    sensor_configs: Dict[str, List[int]],
    durations: List[int],
    batch_size: int = 32,
    n_splits: int = 5,
):
    tasks = [
        TaskSpec(name="lump_binary", label_cols=["Lump"], task_type="binary", num_classes={"Lump": 2}, label_names={"Lump": ["No Lump", "Lump"]}),
        TaskSpec(name="size_multiclass", label_cols=["Size"], task_type="multiclass", num_classes={"Size": 4}, label_names={"Size": ["Small", "Medium", "Big", "No Lump"]}),
        TaskSpec(name="position_multiclass", label_cols=["Position"], task_type="multiclass", num_classes={"Position": 4}, label_names={"Position": ["Top", "Middle", "Bottom", "No Lump"]}),
        TaskSpec(name="multitask_all", label_cols=["Lump", "Size", "Position"], task_type="multitask", num_classes={"Lump": 2, "Size": 4, "Position": 4}, label_names={"Lump": ["No Lump", "Lump"], "Size": ["Small", "Medium", "Big", "No Lump"], "Position": ["Top", "Middle", "Bottom", "No Lump"]}),
    ]

    model_name = "inceptiontime"
    results_path = results_dir / "all_results.csv"
    skip_completed = set()
    prev = read_results_csv(results_path)
    if prev is not None:
        for _, r in prev.iterrows():
            f = safe_int(r.get("fold"))
            d = safe_int(r.get("doctor_trials"))
            sec = safe_int(r.get("num_seconds"))
            sensor_conf = r.get("sensor_config") if pd.notna(r.get("sensor_config")) else "all"
            eval_set = r.get("eval_set") if pd.notna(r.get("eval_set")) else "val"
            if eval_set != "val":
                continue
            if f is None or d is None or sec is None:
                continue
            skip_completed.add((r.get("task"), r.get("model"), r.get("cv_type"), sensor_conf, sec, f, d))

    all_results = []
    for task in tasks:
        for sensor_conf, sensor_indices in sensor_configs.items():
            for duration in durations:
                for cv_type in ["group", "plain"]:
                    fold_results = run_cv(
                        df,
                        model_name,
                        task,
                        cv_type=cv_type,
                        sensor_config=sensor_conf,
                        n_splits=n_splits,
                        batch_size=batch_size,
                        results_dir=results_dir,
                        doctors_df=doctors_df,
                        num_seconds=duration,
                        sensor_indices=sensor_indices,
                        skip_completed=skip_completed,
                        results_path=results_path,
                        external_test_df=doctors_df,
                    )
                    all_results.extend(fold_results)
                    for add_trials in range(1, 16):
                        fold_results = run_cv(
                            df,
                            model_name,
                            task,
                            cv_type=cv_type,
                            sensor_config=sensor_conf,
                            n_splits=n_splits,
                            batch_size=batch_size,
                            results_dir=results_dir,
                            doctors_df=doctors_df,
                            doctor_trials=add_trials,
                            num_seconds=duration,
                            sensor_indices=sensor_indices,
                            skip_completed=skip_completed,
                            results_path=results_path,
                            external_test_df=None,
                        )
                        all_results.extend(fold_results)
    return all_results


def save_results(results: List[Dict], out_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def main(run_heavy: bool = False, run_benchmark: bool = False) -> None:
    data_path = Path("togzhan_data_labeled.pkl")
    doctors_path = Path("doctors_data_labeled.pkl")
    if not data_path.exists() or not doctors_path.exists():
        raise FileNotFoundError("Data files missing. Place togzhan_data_labeled.pkl and doctors_data_labeled.pkl in the workspace directory.")

    df = prepare_dataframe(str(data_path))
    doctors_df = prepare_dataframe(str(doctors_path))
    results_dir = Path("results")
    sensor_configs = {
        "tips": [0, 3, 6, 9, 12],
        "tips_middle": [0, 1, 3, 4, 6, 7, 9, 10, 12, 13],
        "all": list(range(15)),
    }
    durations = [1, 2, 3, 4, 5, 6, 7]

    if run_benchmark:
        benchmark_results = run_benchmark(
            df,
            doctors_df,
            results_dir=results_dir / "benchmark",
            sensor_configs=sensor_configs,
            durations=durations,
        )
        benchmark_csv = results_dir / "benchmark" / "benchmark_results.csv"
        existing = read_results_csv(benchmark_csv)
        combined = pd.concat([existing, pd.DataFrame([normalize_row(r) for r in benchmark_results])], ignore_index=True) if existing is not None else pd.DataFrame([normalize_row(r) for r in benchmark_results])
        combined = combined.drop_duplicates(subset=["task", "model", "cv_type", "eval_set", "sensor_config", "num_seconds", "fold", "doctor_trials"], keep="last")
        combined = combined[ALL_RESULT_COLUMNS]
        benchmark_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(benchmark_csv, index=False)
    else:
        print("Benchmark runs skipped. Set run_benchmark=True in main() to execute benchmarks.")

    if run_heavy:
        full_results = run_full_experiments(
            df,
            doctors_df,
            results_dir=results_dir / "inceptiontime",
            sensor_configs=sensor_configs,
            durations=durations,
        )
        full_csv = results_dir / "inceptiontime" / "all_results.csv"
        existing = read_results_csv(full_csv)
        combined = pd.concat([existing, pd.DataFrame([normalize_row(r) for r in full_results])], ignore_index=True) if existing is not None else pd.DataFrame([normalize_row(r) for r in full_results])
        combined = combined.drop_duplicates(subset=["task", "model", "cv_type", "eval_set", "sensor_config", "num_seconds", "fold", "doctor_trials"], keep="last")
        combined = combined[ALL_RESULT_COLUMNS]
        full_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(full_csv, index=False)
    else:
        print("Heavy experiments skipped. Set run_heavy=True in main() to execute full protocol.")


if __name__ == "__main__":
    main(run_heavy=True, run_benchmark=False)




