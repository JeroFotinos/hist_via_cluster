#!/usr/bin/env python
"""
Two-stage ConvNet pipeline with rotation-invariant loss, Platt calibration,
and Bayesian decision rules for µXRF pixel classification.

Stages:
    1) sample vs not-sample
    2) tumor vs non-tumor (only for sample pixels)

Final output (3-way classification):
    0 -> not sample
    1 -> non-tumoral sample
    2 -> tumoral sample
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from hist_via_cluster.load_dataset import (
    load_pixel_dataframe_with_neighborhood,
    load_fluorescence,
)

# --------------------------------------------------------------------
# Global configuration
# --------------------------------------------------------------------

RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neighborhood size for patches (odd integer: 1,3,5,...)
NEIGHBORHOOD_SIZE = 5
# ConvNet training hyperparameters
BATCH_SIZE = 2048
MAX_EPOCHS_STAGE1 = 16
MAX_EPOCHS_STAGE2 = 19
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


# --------------------------------------------------------------------
# Reproducibility utilities
# --------------------------------------------------------------------

def set_global_seed(seed: int):
    """Set seeds for Python, NumPy and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For strict determinism you could additionally enable:
    # torch.use_deterministic_algorithms(True)


# --------------------------------------------------------------------
# Coarse-graining and label construction
# --------------------------------------------------------------------

def coarse_graining_map(label):
    """
    Map original integer label to coarse-grained codes:
        NaN / 0      -> NaN (no label)
        1            -> 1 (necrotic)    [not present in current data]
        2,3,4        -> 2 (tumoral)
        6,7,9        -> 3 (non-tumoral)
        8,10         -> 4 (no sample / empty)
        5, others    -> NaN
    """
    if pd.isna(label) or label == 0:
        return np.nan
    elif label == 1:
        return 1
    elif label in [2, 3, 4]:
        return 2
    elif label in [6, 7, 9]:
        return 3
    elif label in [8, 10]:
        return 4
    else:
        return np.nan


COARSE_LABEL_NAMES = {
    np.nan: "no label",
    1: "necrótico",
    2: "tumoral",
    3: "no tumoral",
    4: "no muestra",
}


def load_and_prepare_dataframe(data_root: Path, neighborhood_size: int) -> tuple[pd.DataFrame, list[str], dict, int]:
    """
    Load neighborhood-augmented pixel dataframe and construct:
        - new_label (coarse-grained)
        - y_stage1 (sample vs not-sample)
        - y_stage2 (tumor vs non-tumor)
        - feature_cols and col_index_map to reconstruct patches
    """
    assert neighborhood_size % 2 == 1, "neighborhood_size must be an odd integer."

    print("Loading pixel dataframe with neighborhood...")
    df = load_pixel_dataframe_with_neighborhood(
        directory=str(data_root),
        neighborhood_size=neighborhood_size,
    )

    # Coarse-grain labels
    df["new_label"] = df["label"].apply(coarse_graining_map)
    present = sorted(df["new_label"].dropna().unique().tolist())
    print("Coarse-grained categories present (new_label):", present)
    # Typically: [2, 3, 4] (tumoral, non-tumoral, no-sample)

    # Stage 1: sample (2,3) vs not-sample (4)
    df["y_stage1"] = df["new_label"].isin([2, 3]).astype(int)

    # Stage 2: tumor (2) vs non-tumor (3), only meaningful where y_stage1==1
    df["y_stage2"] = (df["new_label"] == 2).astype(int)

    # Identify feature columns
    meta_cols = [
        "diet",
        "mouse",
        "take",
        "row",
        "col",
        "label",
        "new_label",
        "y_stage1",
        "y_stage2",
    ]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    print(f"Number of feature columns: {len(feature_cols)}")

    # Get element order from the original dataset
    ds_dict = load_fluorescence(str(data_root), as_dict=True)
    element_order = ds_dict.element_order
    n_elements = len(element_order)

    k = (neighborhood_size - 1) // 2

    def parse_feature_name(name: str):
        elem, ro, co = name.split("_")
        return elem, int(ro), int(co)

    col_index_map: dict[tuple[int, int, int], int] = {}
    for idx, name in enumerate(feature_cols):
        elem, ro, co = parse_feature_name(name)
        elem_idx = element_order.index(elem)
        col_index_map[(elem_idx, ro, co)] = idx

    # Sanity check: all expected offsets present
    offsets = range(-k, k + 1)
    for elem_idx in range(n_elements):
        for ro in offsets:
            for co in offsets:
                if (elem_idx, ro, co) not in col_index_map:
                    raise RuntimeError(f"Missing feature for (elem_idx={elem_idx}, ro={ro}, co={co})")

    print("Dataframe prepared: new_label, y_stage1, y_stage2, feature_cols, col_index_map ready.")
    return df, feature_cols, col_index_map, n_elements


# --------------------------------------------------------------------
# Splits: train / val / calib / test (with consistent stage 1 & 2 masks)
# --------------------------------------------------------------------

def make_splits(df: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    """
    Create consistent train/val/calib/test pixel-level splits, stratified by new_label.

    Returns a dict of boolean masks:
        train_s1, val_s1, calib_s1, test_s1
        train_s2, val_s2, calib_s2, test_s2
    """
    N = len(df)
    all_idx = np.arange(N)

    # valid = pixels with new_label in {2,3,4}
    valid_mask = df["new_label"].isin([2, 3, 4]).to_numpy()
    idx_valid = all_idx[valid_mask]
    y_cg_valid = df.loc[idx_valid, "new_label"].to_numpy()

    # 1) train+val+calib vs test
    idx_tvc, idx_test = train_test_split(
        idx_valid,
        test_size=0.20,
        random_state=seed,
        stratify=y_cg_valid,
    )

    y_cg_tvc = df.loc[idx_tvc, "new_label"].to_numpy()

    # 2) calib vs train+val
    idx_tv, idx_calib = train_test_split(
        idx_tvc,
        test_size=0.20,  # 20% of 80% -> 16% overall
        random_state=seed,
        stratify=y_cg_tvc,
    )

    y_cg_tv = df.loc[idx_tv, "new_label"].to_numpy()

    # 3) train vs val
    idx_train, idx_val = train_test_split(
        idx_tv,
        test_size=0.25,  # 25% of 64% -> 16% overall
        random_state=seed,
        stratify=y_cg_tv,
    )

    mask_train = np.zeros(N, dtype=bool)
    mask_val = np.zeros(N, dtype=bool)
    mask_calib = np.zeros(N, dtype=bool)
    mask_test = np.zeros(N, dtype=bool)

    mask_train[idx_train] = True
    mask_val[idx_val] = True
    mask_calib[idx_calib] = True
    mask_test[idx_test] = True

    is_sample = df["new_label"].isin([2, 3]).to_numpy()  # Stage 2 only on samples

    # Stage 1 masks: all valid pixels
    mask_train_s1 = mask_train & valid_mask
    mask_val_s1 = mask_val & valid_mask
    mask_calib_s1 = mask_calib & valid_mask
    mask_test_s1 = mask_test & valid_mask

    # Stage 2 masks: only sample pixels
    mask_train_s2 = mask_train & is_sample
    mask_val_s2 = mask_val & is_sample
    mask_calib_s2 = mask_calib & is_sample
    mask_test_s2 = mask_test & is_sample

    print("Split sizes (valid pixels):")
    print("  train :", mask_train_s1.sum())
    print("  val   :", mask_val_s1.sum())
    print("  calib :", mask_calib_s1.sum())
    print("  test  :", mask_test_s1.sum())

    print("Stage 2 sizes (sample pixels only):")
    print("  train :", mask_train_s2.sum())
    print("  val   :", mask_val_s2.sum())
    print("  calib :", mask_calib_s2.sum())
    print("  test  :", mask_test_s2.sum())

    return {
        "train_s1": mask_train_s1,
        "val_s1": mask_val_s1,
        "calib_s1": mask_calib_s1,
        "test_s1": mask_test_s1,
        "train_s2": mask_train_s2,
        "val_s2": mask_val_s2,
        "calib_s2": mask_calib_s2,
        "test_s2": mask_test_s2,
    }


# --------------------------------------------------------------------
# PatchDataset
# --------------------------------------------------------------------

class PatchDataset(Dataset):
    """
    Dataset yielding (patch, label) for binary classification.

    patch: (C, H, W), label: float in {0,1}
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        col_index_map: dict[tuple[int, int, int], int],
        y_column: str,
        mask: np.ndarray,
        n_elements: int,
        neighborhood_size: int,
        dtype=torch.float32,
    ):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.col_index_map = col_index_map
        self.y_column = y_column
        self.indices = np.nonzero(mask)[0]
        self.n_elements = n_elements
        self.k = (neighborhood_size - 1) // 2
        self.size = neighborhood_size
        self.dtype = dtype

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row_idx = self.indices[idx]
        row = self.df.iloc[row_idx]
        x_flat = row[self.feature_cols].to_numpy(dtype=np.float32)

        patch = np.zeros((self.n_elements, self.size, self.size), dtype=np.float32)
        for (elem_idx, ro, co), col_idx in self.col_index_map.items():
            r = self.k - ro  # +ro -> up (decreasing row index)
            c = self.k + co  # +co -> right (increasing col index)
            patch[elem_idx, r, c] = x_flat[col_idx]

        y = float(row[self.y_column])

        x_tensor = torch.from_numpy(patch).to(self.dtype)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return x_tensor, y_tensor


# --------------------------------------------------------------------
# ConvNet with rotation-invariant loss
# --------------------------------------------------------------------

class ConvNetBinary(nn.Module):
    """
    Small ConvNet for binary classification on patches (C,H,W).

    Architecture:
        Conv2d -> ReLU -> Conv2d -> ReLU -> GlobalAvgPool -> FC -> ReLU -> Dropout -> FC(1)
    """

    def __init__(
        self,
        n_channels: int,
        base_channels: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(base_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # logits
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.features(x)
        x = self.avgpool(x)        # (B, C, 1, 1)
        x = torch.flatten(x, 1)    # (B, C)
        logits = self.classifier(x)
        return logits  # (B, 1)


bce_loss = nn.BCEWithLogitsLoss()


def rotation_invariant_bce_loss(model, x, y, criterion=bce_loss, n_rotations: int = 4):
    """
    Average BCE loss over n_rotations (0, 90, 180, 270 degrees).
    """
    total_loss = 0.0
    for k in range(n_rotations):
        x_rot = torch.rot90(x, k=k, dims=(-2, -1))
        logits = model(x_rot).squeeze(1)  # (B,)
        total_loss = total_loss + criterion(logits, y)
    return total_loss / n_rotations


def train_binary_convnet(
    train_dataset: Dataset,
    val_dataset: Dataset,
    n_channels: int,
    max_epochs: int,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    rotation_invariant: bool = True,
    seed: int = RANDOM_STATE,
    device: torch.device = DEVICE,
) -> nn.Module:
    """
    Deterministic training of a ConvNetBinary on a binary task.
    """
    set_global_seed(seed)

    # Fresh generators for shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = ConvNetBinary(n_channels=n_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            if rotation_invariant:
                loss = rotation_invariant_bce_loss(model, xb, yb)
            else:
                logits = model(xb).squeeze(1)
                loss = bce_loss(logits, yb)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        all_y = []
        all_scores = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb).squeeze(1)
                loss = bce_loss(logits, yb)
                val_losses.append(loss.item())

                all_y.append(yb.cpu().numpy())
                all_scores.append(torch.sigmoid(logits).cpu().numpy())

        all_y = np.concatenate(all_y)
        all_scores = np.concatenate(all_scores).ravel()
        val_auc = roc_auc_score(all_y, all_scores)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={np.mean(train_losses):.4f} "
            f"val_loss={np.mean(val_losses):.4f} "
            f"val_auc={val_auc:.4f}"
        )

    return model


# --------------------------------------------------------------------
# Calibration (Platt scaling)
# --------------------------------------------------------------------

def collect_logits_and_labels(model: nn.Module, dataset: Dataset, batch_size: int = BATCH_SIZE) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_logits = []
    all_y = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb).squeeze(1)
            all_logits.append(logits.cpu().numpy())
            all_y.append(yb.cpu().numpy())
    return np.concatenate(all_logits), np.concatenate(all_y)


def fit_platt_scaler(logits: np.ndarray, y: np.ndarray, seed: int = RANDOM_STATE) -> LogisticRegression:
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    clf.fit(logits.reshape(-1, 1), y)
    return clf


def calibrated_proba(model: nn.Module, calibrator: LogisticRegression, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = x.to(DEVICE)
        logits = model(x).squeeze(1)
        logits_np = logits.cpu().numpy().reshape(-1, 1)
    return calibrator.predict_proba(logits_np)[:, 1]


# --------------------------------------------------------------------
# Bayesian decision rule
# --------------------------------------------------------------------

class BayesianDecisionRule:
    """
    Bayes-optimal decision rule with asymmetric costs:
        C10 = cost(false positive),
        C01 = cost(false negative)

    threshold = C10 / (C10 + C01)
    """

    def __init__(self, C10: float = 1.0, C01: float = 1.0):
        self.C10 = float(C10)
        self.C01 = float(C01)
        self.threshold = self.C10 / (self.C10 + self.C01)

    def predict(self, proba: np.ndarray) -> np.ndarray:
        proba = np.asarray(proba)
        return (proba >= self.threshold).astype(int)

    def __repr__(self):
        return f"BayesianDecisionRule(C10={self.C10}, C01={self.C01}, threshold={self.threshold:.3f})"


# --------------------------------------------------------------------
# Two-stage prediction on a DataFrame subset
# --------------------------------------------------------------------

def build_patch_tensor_for_rows(
    df_subset: pd.DataFrame,
    feature_cols: list[str],
    col_index_map: dict[tuple[int, int, int], int],
    n_elements: int,
    neighborhood_size: int,
) -> torch.Tensor:
    k = (neighborhood_size - 1) // 2
    size = neighborhood_size
    N = len(df_subset)

    patches = np.zeros((N, n_elements, size, size), dtype=np.float32)
    X = df_subset[feature_cols].to_numpy(dtype=np.float32)

    for i in range(N):
        x_flat = X[i]
        patch = patches[i]
        for (elem_idx, ro, co), col_idx in col_index_map.items():
            r = k - ro
            c = k + co
            patch[elem_idx, r, c] = x_flat[col_idx]

    return torch.from_numpy(patches).float()


def two_stage_predict_df(
    df_subset: pd.DataFrame,
    feature_cols: list[str],
    col_index_map: dict[tuple[int, int, int], int],
    model_stage1: nn.Module,
    calibrator1: LogisticRegression,
    rule_stage1: BayesianDecisionRule,
    model_stage2: nn.Module,
    calibrator2: LogisticRegression,
    rule_stage2: BayesianDecisionRule,
    n_elements: int,
    neighborhood_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the full two-stage pipeline on df_subset.

    Returns:
        final_labels: {0,1,2}
            0 -> not sample
            1 -> non-tumoral
            2 -> tumoral
        p_sample: P(sample | x)
        p_tumor:  P(tumor | x, sample)
    """
    x_tensor = build_patch_tensor_for_rows(
        df_subset,
        feature_cols,
        col_index_map,
        n_elements=n_elements,
        neighborhood_size=neighborhood_size,
    )

    # Stage 1: sample vs not-sample
    p_sample = calibrated_proba(model_stage1, calibrator1, x_tensor)
    yhat_sample = rule_stage1.predict(p_sample)

    # Stage 2: tumor vs non-tumor for predicted samples
    p_tumor = np.zeros_like(p_sample)
    yhat_tumor = np.zeros_like(yhat_sample)

    is_sample = (yhat_sample == 1)
    if is_sample.any():
        x_sample = x_tensor[is_sample]
        p_tumor_sample = calibrated_proba(model_stage2, calibrator2, x_sample)
        yhat_tumor_sample = rule_stage2.predict(p_tumor_sample)
        p_tumor[is_sample] = p_tumor_sample
        yhat_tumor[is_sample] = yhat_tumor_sample

    # Final 3-class labels
    final = np.zeros_like(yhat_sample)
    final[(is_sample) & (yhat_tumor == 0)] = 1  # non-tumor sample
    final[(is_sample) & (yhat_tumor == 1)] = 2  # tumor sample

    return final, p_sample, p_tumor


# --------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data"

    print("Project root:", project_root)
    print("Data root   :", data_root)

    set_global_seed(RANDOM_STATE)

    # 1) Load and prepare df + feature mapping
    df, feature_cols, col_index_map, n_elements = load_and_prepare_dataframe(
        data_root=data_root,
        neighborhood_size=NEIGHBORHOOD_SIZE,
    )

    # 2) Build splits and datasets
    masks = make_splits(df, seed=RANDOM_STATE)

    train_ds1 = PatchDataset(
        df=df,
        feature_cols=feature_cols,
        col_index_map=col_index_map,
        y_column="y_stage1",
        mask=masks["train_s1"],
        n_elements=n_elements,
        neighborhood_size=NEIGHBORHOOD_SIZE,
    )
    val_ds1 = PatchDataset(
        df=df,
        feature_cols=feature_cols,
        col_index_map=col_index_map,
        y_column="y_stage1",
        mask=masks["val_s1"],
        n_elements=n_elements,
        neighborhood_size=NEIGHBORHOOD_SIZE,
    )
    calib_ds1 = PatchDataset(
        df=df,
        feature_cols=feature_cols,
        col_index_map=col_index_map,
        y_column="y_stage1",
        mask=masks["calib_s1"],
        n_elements=n_elements,
        neighborhood_size=NEIGHBORHOOD_SIZE,
    )
    test_ds1 = PatchDataset(
        df=df,
        feature_cols=feature_cols,
        col_index_map=col_index_map,
        y_column="y_stage1",
        mask=masks["test_s1"],
        n_elements=n_elements,
        neighborhood_size=NEIGHBORHOOD_SIZE,
    )

    train_ds2 = PatchDataset(
        df=df,
        feature_cols=feature_cols,
        col_index_map=col_index_map,
        y_column="y_stage2",
        mask=masks["train_s2"],
        n_elements=n_elements,
        neighborhood_size=NEIGHBORHOOD_SIZE,
    )
    val_ds2 = PatchDataset(
        df=df,
        feature_cols=feature_cols,
        col_index_map=col_index_map,
        y_column="y_stage2",
        mask=masks["val_s2"],
        n_elements=n_elements,
        neighborhood_size=NEIGHBORHOOD_SIZE,
    )
    calib_ds2 = PatchDataset(
        df=df,
        feature_cols=feature_cols,
        col_index_map=col_index_map,
        y_column="y_stage2",
        mask=masks["calib_s2"],
        n_elements=n_elements,
        neighborhood_size=NEIGHBORHOOD_SIZE,
    )
    test_ds2 = PatchDataset(
        df=df,
        feature_cols=feature_cols,
        col_index_map=col_index_map,
        y_column="y_stage2",
        mask=masks["test_s2"],
        n_elements=n_elements,
        neighborhood_size=NEIGHBORHOOD_SIZE,
    )

    # 3) Train ConvNets for both stages
    print("\n=== Training Stage 1 (sample vs not-sample) ===")
    model_stage1 = train_binary_convnet(
        train_dataset=train_ds1,
        val_dataset=val_ds1,
        n_channels=n_elements,
        max_epochs=MAX_EPOCHS_STAGE1,
    )

    print("\n=== Training Stage 2 (tumor vs non-tumor) ===")
    model_stage2 = train_binary_convnet(
        train_dataset=train_ds2,
        val_dataset=val_ds2,
        n_channels=n_elements,
        max_epochs=MAX_EPOCHS_STAGE2,
    )

    # 4) Calibration
    print("\n=== Calibrating Stage 1 ===")
    logits_calib1, y_calib1 = collect_logits_and_labels(model_stage1, calib_ds1)
    calibrator1 = fit_platt_scaler(logits_calib1, y_calib1)

    print("\n=== Calibrating Stage 2 ===")
    logits_calib2, y_calib2 = collect_logits_and_labels(model_stage2, calib_ds2)
    calibrator2 = fit_platt_scaler(logits_calib2, y_calib2)

    # 5) Bayesian decision rules (tunable costs)
    rule_stage1 = BayesianDecisionRule(C10=1.0, C01=1.0)  # balanced
    rule_stage2 = BayesianDecisionRule(C10=1.0, C01=5.0)  # penalize missed tumors

    print("\nDecision rules:")
    print("  Stage 1:", rule_stage1)
    print("  Stage 2:", rule_stage2)

    # 6) Final evaluation on test split (3-way)
    print("\n=== Final Evaluation on Test Set (3-way) ===")
    mask_test = masks["test_s1"]  # all valid pixels in test

    df_test = df.loc[mask_test].copy()
    true_new_label = df_test["new_label"].to_numpy()

    # Map new_label -> {0,1,2}
    # 4 -> 0 (no muestra / not sample)
    # 3 -> 1 (no tumoral)
    # 2 -> 2 (tumoral)
    true_final = np.zeros_like(true_new_label, dtype=int)
    true_final[true_new_label == 3] = 1
    true_final[true_new_label == 2] = 2

    final_pred, p_sample_test, p_tumor_test = two_stage_predict_df(
        df_subset=df_test,
        feature_cols=feature_cols,
        col_index_map=col_index_map,
        model_stage1=model_stage1,
        calibrator1=calibrator1,
        rule_stage1=rule_stage1,
        model_stage2=model_stage2,
        calibrator2=calibrator2,
        rule_stage2=rule_stage2,
        n_elements=n_elements,
        neighborhood_size=NEIGHBORHOOD_SIZE,
    )

    print(classification_report(true_final, final_pred, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(true_final, final_pred))


if __name__ == "__main__":
    main()
