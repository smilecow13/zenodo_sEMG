"""
Chọn lọc đặc trưng TRONG fold (mục 5.4, tránh leakage)
=======================================================
Chọn feature trên toàn bộ dữ liệu rồi mới chạy LOSO là leakage: tập test đã
tham gia vào việc quyết định dùng feature nào, nên RMSE báo ra bị thổi phồng.
Đây là dạng lỗi reviewer bắt rất nhanh, và cùng họ với lỗi early-stopping
trên test set đã xảy ra ở WP2.

Cách đúng: trong mỗi fold LOSO, chọn feature CHỈ từ 12 subject train rồi áp
lên subject test. Số feature và danh sách feature được chọn khác nhau giữa
các fold, và chính sự khác nhau đó là thông tin đáng báo cáo — feature nào
sống sót ở cả 13 fold mới là feature ổn định để nạp vào firmware.

Module cũng cung cấp phiên bản chọn TOÀN CỤC (`run_loso_global_selection`)
nhưng chỉ để ĐỊNH LƯỢNG mức thổi phồng do leakage, không dùng để báo cáo
kết quả. Hiệu số giữa hai phiên bản là một con số đáng đưa vào bài.

Ghi chú toán học có ích: trong cùng một trial, phép z-score neo
x ↦ (x − μ)/σ với σ > 0 là biến đổi đơn điệu tăng, nên Spearman tính
TRONG TRIAL cho cột thô và cột _N1 là GIỐNG HỆT NHAU. Vì vậy xếp hạng theo
rho per-trial không phụ thuộc việc dùng biểu diễn thô hay đã chuẩn hoá.

Usage:
    from feature_selection import run_loso_in_fold_selection, rank_by_per_trial_rho

    res, log = run_loso_in_fold_selection(
        RandomForestRegressor, candidates, 'FCF', df, k=12, model_kwargs=RF_KWARGS)
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_protocols import TARGET_SPECS, _fit_predict, _score

MIN_REPS_PER_TRIAL = 10


# ═══════════════════════════════════════════════════════════════════════════
# Xếp hạng theo độ liên quan
# ═══════════════════════════════════════════════════════════════════════════
def rank_by_per_trial_rho(df, cols, target="FCF", min_reps=MIN_REPS_PER_TRIAL):
    """
    Xếp hạng feature theo |trung vị Spearman rho per-trial| giảm dần.

    Tính trong từng (subject, trial) rồi lấy trung vị: gộp thô mọi trial lại
    sẽ lẫn biến thiên gain giữa các subject và làm loãng tương quan thật
    (MDF gộp thô cho rho = −0.04, trong từng trial lại đạt −0.40).

    Returns
    -------
    pd.Series : index = tên feature, value = |rho trung vị|, sắp giảm dần
    """
    groups = [g for _, g in df.groupby(["subject", "trial"])]
    scores = {}

    for col in cols:
        if col not in df.columns:
            continue
        rhos = []
        for grp in groups:
            if grp[col].notna().sum() < min_reps or grp[col].nunique() < 3:
                continue
            rho, _ = spearmanr(grp[col], grp[target], nan_policy="omit")
            if np.isfinite(rho):
                rhos.append(rho)
        if rhos:
            scores[col] = abs(float(np.median(rhos)))

    return pd.Series(scores).sort_values(ascending=False)


def select_top_k_rho(df_train, cols, k, target="FCF"):
    """Chọn k feature có |rho trung vị per-trial| cao nhất, tính trên train."""
    ranked = rank_by_per_trial_rho(df_train, cols, target=target)
    return list(ranked.head(k).index)


def select_mrmr(df_train, cols, k, target="FCF", redundancy_weight=1.0):
    """
    mRMR tham lam: mỗi bước chọn feature tối đa (độ liên quan − độ dư thừa).

    Độ liên quan = |rho trung vị per-trial| với target.
    Độ dư thừa   = trung bình |Pearson| với các feature ĐÃ chọn.

    Feature đầu tiên là feature liên quan nhất; từ bước thứ hai trở đi mới
    có thành phần dư thừa. Dùng Pearson cho dư thừa (không phải Spearman) vì
    đây là bước loại cộng tuyến, cùng thang đo với `drop_collinear`.
    """
    relevance = rank_by_per_trial_rho(df_train, cols, target=target)
    if relevance.empty:
        return []

    available = [c for c in relevance.index if c in df_train.columns]
    corr = df_train[available].corr().abs()

    selected = [available[0]]
    while len(selected) < min(k, len(available)):
        best_col, best_score = None, -np.inf
        for col in available:
            if col in selected:
                continue
            redundancy = float(corr.loc[col, selected].mean())
            score = relevance[col] - redundancy_weight * redundancy
            if score > best_score:
                best_col, best_score = col, score
        if best_col is None:
            break
        selected.append(best_col)

    return selected


# ═══════════════════════════════════════════════════════════════════════════
# LOSO có chọn feature trong fold
# ═══════════════════════════════════════════════════════════════════════════
def run_loso_in_fold_selection(
    model_cls,
    candidate_cols,
    target_col,
    df,
    k,
    selector=select_top_k_rho,
    model_kwargs=None,
    model_needs_scale=False,
    verbose=False,
):
    """
    LOSO với chọn feature thực hiện RIÊNG trong mỗi fold, chỉ từ train.

    Dùng lại `_fit_predict` và `_score` của eval_protocols để kết quả so sánh
    trực tiếp được với A1/A4 (cùng cách clip y_range và cùng cách nhân 100).

    Parameters
    ----------
    candidate_cols : list[str]
        Tập ứng viên để chọn ra k feature.
    k : int
        Số feature giữ lại mỗi fold.
    selector : callable(df_train, cols, k, target) -> list[str]

    Returns
    -------
    results : pd.DataFrame   (test_subject, rmse, mae, r2, n_features)
    log : pd.DataFrame       (feature, n_folds_selected, folds) — độ ổn định
    """
    if model_kwargs is None:
        model_kwargs = {}

    spec = TARGET_SPECS.get(target_col, TARGET_SPECS["FCF"])
    subjects = sorted(df["subject"].unique())
    candidates = [c for c in candidate_cols if c in df.columns]

    results = []
    chosen_per_fold = {}

    for test_subj in subjects:
        train = df[df["subject"] != test_subj]
        test = df[df["subject"] == test_subj]
        if len(test) < 3:
            continue

        feats = selector(train, candidates, k, target_col)
        if not feats:
            continue
        chosen_per_fold[test_subj] = feats

        y_pred = _fit_predict(
            model_cls,
            model_kwargs,
            train[feats].values,
            train[target_col].values,
            test[feats].values,
            y_range=spec["y_range"],
            needs_scale=model_needs_scale,
        )
        scores = _score(test[target_col].values, y_pred, scale_pct=spec["scale_pct"])
        scores["test_subject"] = test_subj
        scores["n_features"] = len(feats)
        results.append(scores)

        if verbose:
            print(f"    fold S{test_subj}: {feats[:5]}{'...' if len(feats) > 5 else ''}")

    # Độ ổn định: feature nào được chọn ở bao nhiêu fold
    counts = {}
    for subj, feats in chosen_per_fold.items():
        for f in feats:
            counts.setdefault(f, []).append(subj)

    log = pd.DataFrame(
        [
            {"feature": f, "n_folds_selected": len(s), "folds": ",".join(map(str, s))}
            for f, s in counts.items()
        ]
    ).sort_values("n_folds_selected", ascending=False)

    return pd.DataFrame(results), log


def run_loso_global_selection(
    model_cls,
    candidate_cols,
    target_col,
    df,
    k,
    selector=select_top_k_rho,
    model_kwargs=None,
    model_needs_scale=False,
):
    """
    CHỈ ĐỂ ĐO MỨC THỔI PHỒNG DO LEAKAGE — không dùng để báo cáo kết quả.

    Chọn feature một lần trên TOÀN BỘ dữ liệu (gồm cả test) rồi mới chạy LOSO.
    Hiệu số RMSE so với `run_loso_in_fold_selection` chính là phần bị thổi
    phồng khi chọn feature ngoài vòng CV.
    """
    from eval_protocols import run_loso

    feats = selector(df, [c for c in candidate_cols if c in df.columns], k, target_col)
    res = run_loso(
        model_cls, feats, target_col, df,
        model_kwargs=model_kwargs, model_needs_scale=model_needs_scale,
    )
    res["n_features"] = len(feats)
    return res, feats
