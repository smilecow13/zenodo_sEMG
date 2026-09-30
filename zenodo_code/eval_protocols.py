"""
Evaluation Protocols Module
============================
Bộ giao thức đánh giá Cross-Validation cho dự án sEMG FCF/RIR.

Protocols:
  1. LOSO  — Leave-One-Subject-Out (đã có, refactored)
  2. LOMO  — Leave-One-Muscle-Out (MỚI — trên Zenodo, muscle ≈ exercise)
  3. LOSSO — Leave-One-Session-Out (STUB cho WP4 — Zenodo chỉ có 1 session/subject)

⚠️ LOEO vs LOMO:
    Trên Zenodo, "exercise" thực chất là các nhóm cơ khác nhau của cùng kiểu
    vận động (tất cả đều là lateral raise / curl), KHÔNG phải bài tập khác nhau
    về sinh cơ học (như Bicep Curl vs Leg Extension sẽ có ở WP4).
    → Đặt tên LOMO (Leave-One-Muscle-Out) cho Zenodo.
    → Khi có WP4 data (nhiều bài tập thật sự), mới chuyển sang LOEO.

Usage:
    from eval_protocols import run_loso, run_lomo, run_losso
    results = run_loso(model_cls, feat_cols, target_col, df, model_kwargs)
"""

import os
import hashlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ── Target specifications ────────────────────────────────────────────────────
TARGET_SPECS = {
    "FCF": {
        "y_range": (0.0, 1.0),
        "scale_pct": True,  # Report as percentage
        "label": "FCF (%)",
    },
    "RIR_clipped": {
        "y_range": (0.0, 10.0),
        "scale_pct": False,  # Report as rep count
        "label": "RIR (reps)",
    },
}


# ── Core scoring ─────────────────────────────────────────────────────────────
def _score(y_true, y_pred, scale_pct=True):
    """
    Tính RMSE, MAE, R² cho một fold.

    Parameters
    ----------
    scale_pct : bool
        Nếu True, nhân RMSE/MAE × 100 (cho FCF ∈ [0,1] → %)
    """
    factor = 100.0 if scale_pct else 1.0
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * factor
    mae = mean_absolute_error(y_true, y_pred) * factor
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _fit_predict(
    model_cls, model_kwargs, X_train, y_train, X_test, y_range=(0, 1), needs_scale=False
):
    """
    Fit model và predict, clip output vào y_range.
    """
    model = model_cls(**model_kwargs)

    if needs_scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, y_range[0], y_range[1])
    return y_pred


# ── LOSO (Leave-One-Subject-Out) ─────────────────────────────────────────────
def run_loso(
    model_cls, feat_cols, target_col, df, model_kwargs=None, model_needs_scale=False
):
    """
    Leave-One-Subject-Out Cross-Validation.

    13-fold CV trên Zenodo: train trên 12 subjects, test trên 1.
    Zero subject-identity leakage.

    Parameters
    ----------
    model_cls : sklearn estimator class
    feat_cols : list of str
    target_col : str ('FCF' or 'RIR_clipped')
    df : pd.DataFrame
    model_kwargs : dict
    model_needs_scale : bool
        True cho linear models cần StandardScaler

    Returns
    -------
    pd.DataFrame
        Columns: test_subject, rmse, mae, r2
    """
    if model_kwargs is None:
        model_kwargs = {}

    spec = TARGET_SPECS.get(target_col, TARGET_SPECS["FCF"])
    subjects = sorted(df["subject"].unique())
    results = []

    for test_subj in subjects:
        train_mask = df["subject"] != test_subj
        test_mask = df["subject"] == test_subj

        X_train = df.loc[train_mask, feat_cols].values
        y_train = df.loc[train_mask, target_col].values
        X_test = df.loc[test_mask, feat_cols].values
        y_test = df.loc[test_mask, target_col].values

        y_pred = _fit_predict(
            model_cls,
            model_kwargs,
            X_train,
            y_train,
            X_test,
            y_range=spec["y_range"],
            needs_scale=model_needs_scale,
        )

        scores = _score(y_test, y_pred, scale_pct=spec["scale_pct"])
        scores["test_subject"] = test_subj
        results.append(scores)

    return pd.DataFrame(results)


# ── LOMO (Leave-One-Muscle-Out) ──────────────────────────────────────────────
def run_lomo(
    model_cls,
    feat_cols,
    target_col,
    df,
    model_kwargs=None,
    model_needs_scale=False,
    muscle_col="muscle",
):
    """
    Leave-One-Muscle-Out Cross-Validation.

    ⚠️ CHÚ Ý QUAN TRỌNG CHO PAPER:
    Trên Zenodo dataset, các "exercises" thực chất là các nhóm cơ khác nhau
    (Deltoid Anterior, Deltoid Posterior, Biceps Brachii, Deltoid Medius)
    thực hiện cùng kiểu vận động (lateral raises, curls).
    Đây KHÔNG phải là Leave-One-Exercise-Out (LOEO) đúng nghĩa vì:
    - Các bài tập không khác nhau về sinh cơ học
    - Đều là chi trên, không có chi dưới
    - LOEO thật sự cần WP4 data (Bicep Curl vs Leg Extension)

    Khi trình bày trong paper, phải gọi đúng tên "LOMO" hoặc ghi chú rõ
    "leave-one-muscle-group-out as proxy for LOEO" để tránh reviewer phản biện.

    Parameters
    ----------
    model_cls : sklearn estimator class
    feat_cols : list of str
    target_col : str
    df : pd.DataFrame
    model_kwargs : dict
    model_needs_scale : bool
    muscle_col : str
        Cột chứa tên muscle/exercise (default: 'muscle')

    Returns
    -------
    pd.DataFrame
        Columns: test_muscle, rmse, mae, r2, n_test_reps
    """
    if model_kwargs is None:
        model_kwargs = {}

    spec = TARGET_SPECS.get(target_col, TARGET_SPECS["FCF"])
    muscles = sorted(df[muscle_col].unique())
    results = []

    for test_muscle in muscles:
        train_mask = df[muscle_col] != test_muscle
        test_mask = df[muscle_col] == test_muscle

        X_train = df.loc[train_mask, feat_cols].values
        y_train = df.loc[train_mask, target_col].values
        X_test = df.loc[test_mask, feat_cols].values
        y_test = df.loc[test_mask, target_col].values

        if len(y_test) < 10:
            print(f"  [LOMO] Skipping {test_muscle}: only {len(y_test)} reps")
            continue

        y_pred = _fit_predict(
            model_cls,
            model_kwargs,
            X_train,
            y_train,
            X_test,
            y_range=spec["y_range"],
            needs_scale=model_needs_scale,
        )

        scores = _score(y_test, y_pred, scale_pct=spec["scale_pct"])
        scores["test_muscle"] = test_muscle
        scores["n_test_reps"] = len(y_test)
        results.append(scores)

    return pd.DataFrame(results)


# ── LOSSO (Leave-One-Session-Out) ────────────────────────────────────────────
def run_losso(
    model_cls,
    feat_cols,
    target_col,
    df,
    model_kwargs=None,
    model_needs_scale=False,
    session_col="session_id",
):
    """
    Leave-One-Session-Out Cross-Validation.

    ⚠️ STUB — Chỉ chạy được khi có WP4 data (2+ sessions/subject).
    Zenodo chỉ có 1 session/subject → LOSSO KHÔNG ÁP DỤNG được.

    Mục đích: đo độ bền với việc dán lại điện cực khác buổi.
    Thầy nhấn mạnh: "rất ít bài làm, làm được là điểm cộng lớn."

    Parameters
    ----------
    session_col : str
        Tên cột session identifier trong df (phải có ≥2 unique values)

    Returns
    -------
    pd.DataFrame or None
        None nếu data không có đủ sessions
    """
    if model_kwargs is None:
        model_kwargs = {}

    if session_col not in df.columns:
        print(
            f"  [LOSSO] Column '{session_col}' not found. "
            "LOSSO requires multi-session data (WP4)."
        )
        return None

    sessions = sorted(df[session_col].unique())
    if len(sessions) < 2:
        print(
            f"  [LOSSO] Only {len(sessions)} session(s) found. "
            "Need ≥2 sessions. LOSSO not applicable for this dataset."
        )
        return None

    spec = TARGET_SPECS.get(target_col, TARGET_SPECS["FCF"])
    results = []

    for test_session in sessions:
        train_mask = df[session_col] != test_session
        test_mask = df[session_col] == test_session

        X_train = df.loc[train_mask, feat_cols].values
        y_train = df.loc[train_mask, target_col].values
        X_test = df.loc[test_mask, feat_cols].values
        y_test = df.loc[test_mask, target_col].values

        y_pred = _fit_predict(
            model_cls,
            model_kwargs,
            X_train,
            y_train,
            X_test,
            y_range=spec["y_range"],
            needs_scale=model_needs_scale,
        )

        scores = _score(y_test, y_pred, scale_pct=spec["scale_pct"])
        scores["test_session"] = test_session
        scores["n_test_reps"] = len(y_test)
        results.append(scores)

    return pd.DataFrame(results)


# ── Near-Failure Evaluation (existing logic, refactored) ─────────────────────
def run_loso_near_failure(
    model_cls,
    feat_cols,
    target_col,
    df,
    model_kwargs=None,
    rir_threshold=10,
    model_needs_scale=False,
):
    """
    LOSO nhưng chỉ đánh giá trên near-failure zone (RIR < threshold).
    Train trên TẤT CẢ reps, test chỉ trên near-failure reps.

    Quan trọng vì 86.9% reps trong Zenodo có RIR_clipped=10 (ceiling effect).
    """
    if model_kwargs is None:
        model_kwargs = {}

    spec = TARGET_SPECS.get(target_col, TARGET_SPECS["FCF"])
    subjects = sorted(df["subject"].unique())
    results = []

    for test_subj in subjects:
        train_mask = df["subject"] != test_subj
        test_mask = (df["subject"] == test_subj) & (df[target_col] < rir_threshold)

        if test_mask.sum() == 0:
            continue

        X_train = df.loc[train_mask, feat_cols].values
        y_train = df.loc[train_mask, target_col].values
        X_test = df.loc[test_mask, feat_cols].values
        y_test = df.loc[test_mask, target_col].values

        y_pred = _fit_predict(
            model_cls,
            model_kwargs,
            X_train,
            y_train,
            X_test,
            y_range=spec["y_range"],
            needs_scale=model_needs_scale,
        )

        scores = _score(y_test, y_pred, scale_pct=spec["scale_pct"])
        scores["test_subject"] = test_subj
        scores["n_near_failure"] = len(y_test)
        results.append(scores)

    return pd.DataFrame(results)


def run_loso_near_failure_train_filtered(
    model_cls,
    feat_cols,
    target_col,
    df,
    model_kwargs=None,
    rir_threshold=10,
    model_needs_scale=False,
    train_fcf_min=None,
):
    """
    Giống run_loso_near_failure, nhưng cho phép lọc RIÊNG training set theo FCF
    (loại phần đầu set 'luôn RIR=10' khỏi training), trong khi TEST SET giữ
    nguyên toàn bộ near-failure (RIR<10, không phân biệt FCF) để so sánh
    công bằng với baseline gốc.
    """
    if model_kwargs is None:
        model_kwargs = {}
    spec = TARGET_SPECS.get(target_col, TARGET_SPECS["FCF"])
    subjects = sorted(df["subject"].unique())
    results = []

    for test_subj in subjects:
        train_mask = df["subject"] != test_subj
        if train_fcf_min is not None:
            train_mask &= df["FCF"] > train_fcf_min  # CHỈ lọc train
        test_mask = (df["subject"] == test_subj) & (
            df[target_col] < rir_threshold
        )  # test giữ nguyên

        X_train = df.loc[train_mask, feat_cols].values
        y_train = df.loc[train_mask, target_col].values
        X_test = df.loc[test_mask, feat_cols].values
        y_test = df.loc[test_mask, target_col].values

        if len(y_test) == 0:
            continue

        y_pred = _fit_predict(
            model_cls,
            model_kwargs,
            X_train,
            y_train,
            X_test,
            y_range=spec["y_range"],
            needs_scale=model_needs_scale,
        )
        scores = _score(y_test, y_pred, scale_pct=spec["scale_pct"])
        scores["test_subject"] = test_subj
        scores["n_near_failure"] = len(y_test)
        results.append(scores)

    return pd.DataFrame(results)


# ── Summary & Comparison Utilities ───────────────────────────────────────────
def summary(results_df, metric_cols=None):
    """Tính mean ± std của các metric qua các fold."""
    if metric_cols is None:
        metric_cols = ["rmse", "mae", "r2"]
    out = {}
    for col in metric_cols:
        if col in results_df.columns:
            out[f"{col}_mean"] = results_df[col].mean()
            out[f"{col}_std"] = results_df[col].std()
    return out


def compare_paired(
    res_a, res_b, label_a="A", label_b="B", merge_key="test_subject", alpha=0.05
):
    """
    Wilcoxon signed-rank test giữa 2 configuration.

    Parameters
    ----------
    res_a, res_b : pd.DataFrame
        Kết quả LOSO/LOMO (phải có cột 'rmse' và merge_key)
    label_a, label_b : str
    merge_key : str
    alpha : float

    Returns
    -------
    dict with delta_mean, delta_std, p_value, significant
    """
    from scipy.stats import wilcoxon

    merged = res_a.merge(res_b, on=merge_key, suffixes=("_a", "_b"))
    delta = merged["rmse_a"] - merged["rmse_b"]
    delta_mean = delta.mean()
    delta_std = delta.std()

    if len(delta) < 6:
        print(f"  {label_a} vs {label_b}: too few folds ({len(delta)}) for Wilcoxon")
        return {
            "delta_mean": delta_mean,
            "delta_std": delta_std,
            "p_value": float("nan"),
            "significant": False,
        }

    stat, p_value = wilcoxon(merged["rmse_a"], merged["rmse_b"])
    sig = p_value < alpha
    sig_label = "SIG" if sig else "n.s."

    print(
        f"  {label_a} vs {label_b}: "
        f"delta={delta_mean:+.2f}pp ± {delta_std:.2f}, "
        f"p={p_value:.4f} [{sig_label}]"
    )

    return {
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "p_value": p_value,
        "significant": sig,
    }


def check_go_no_go(rmse_f0, rmse_f1, model_name, threshold=0.15):
    """
    Go/No-Go decision: relative improvement ≥ threshold.

    ⚠️ Đây là ngưỡng Go/No-Go THÁNG 4 (relative improvement),
    KHÔNG PHẢI mục tiêu RMSE tuyệt đối < 15% (cuối kỳ).
    """
    improvement = (rmse_f0 - rmse_f1) / rmse_f0
    decision = "GO" if improvement >= threshold else "NO-GO"
    print(
        f"  {model_name}: F0={rmse_f0:.2f}% → F1={rmse_f1:.2f}% | "
        f"improvement={improvement * 100:.1f}% [{decision}]"
    )
    return improvement >= threshold


if __name__ == "__main__":
    print("Evaluation protocols module loaded successfully.")
    print(
        f"Available protocols: run_loso, run_lomo, run_losso, run_loso_near_failure, run_loso_near_failure_train_filtered"
    )
    print(f"Target specs: {list(TARGET_SPECS.keys())}")
