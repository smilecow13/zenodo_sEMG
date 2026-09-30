"""
N1 Feature Extension Module
============================
Mở rộng hệ thống chuẩn hóa N1 theo đặc tả thầy hướng dẫn.

Ba biến thể chuẩn hóa neo theo 3 rep đầu mỗi (subject, trial):
  1. Z-score:     x̃_k = (x_k − μ_{1:3}) / σ_{1:3}      [ĐÃ CÓ trong notebook]
  2. Ratio:       r_k  = x_k / x̄_{1:3}                   [MỚI]
  3. Differential: Δx_k = x_k − x_{k-1}                   [MỚI]

Usage:
    from n1_features import build_all_feature_sets
    feature_sets, df = build_all_feature_sets(df)
"""

import numpy as np
import pandas as pd


# ── Epsilon TƯƠNG ĐỐI cho mẫu số z-score ────────────────────────────────────
# Epsilon TUYỆT ĐỐI (kiểu `sg + 1e-9`) là bug âm thầm khi feature có độ lớn
# nhỏ: FInsm5 có σ baseline ~2.2e-13, nên 1e-9 lấn át σ thật 4500 lần và
# z-score ra ~1e-4 thay vì ~±0.24. Không NaN, không lỗi chạy — chỉ âm thầm
# biến "chuẩn hoá phương sai" thành "chỉ trừ trung bình", tức mất đúng phần
# mang lại +12.78% ở A1. Dùng sàn TƯƠNG ĐỐI theo thang của chính feature.
REL_EPS = 1e-9


# ── Feature column definitions ───────────────────────────────────────────────
# 12 raw features (F0) — same as notebook
F0_COLS = [
    "MNF_max",
    "MNF_min",
    "MNF_mean",
    "MDF_max",
    "MDF_min",
    "MDF_mean",
    "TP_max",
    "TP_min",
    "TP_mean",
    "RMS_max",
    "RMS_min",
    "RMS_mean",
]

# 12 z-score features (existing N1)
F1_ZSCORE_COLS = [f"{c}_N1" for c in F0_COLS]

# 12 ratio features (new)
F1_RATIO_COLS = [f"{c}_ratio" for c in F0_COLS]

# 12 differential features (new)
F1_DIFF_COLS = [f"{c}_diff" for c in F0_COLS]


def apply_N1_zscore(df, cols=None, baseline_n=3, suffix="_N1", rel_eps=REL_EPS, report=True):
    """
    N1 z-score neo theo `baseline_n` rep đầu mỗi (subject, trial):

        x̂_k = (x_k − μ_{1:n}) / σ_{1:n}

    Bản canonical của công thức N1 — thay cho `apply_N1_normalization` trong
    notebook (hàm đó khoá cứng 12 cột F0 và dùng epsilon tuyệt đối 1e-9).

    Khác biệt then chốt: mẫu số dùng SÀN TƯƠNG ĐỐI `rel_eps · |μ|` thay vì
    cộng thêm một epsilon tuyệt đối. Nhờ vậy feature có độ lớn bất kỳ đều
    được chuẩn hoá phương sai đúng (xem chú thích ở REL_EPS). Sàn chỉ kích
    hoạt khi σ thực sự suy biến, và số lần kích hoạt được báo ra thay vì
    ẩn đi, để không lặp lại kiểu lỗi âm thầm như epsilon tuyệt đối.

    Parameters
    ----------
    df : pd.DataFrame
        Phải có subject, trial, rep_idx và các cột trong `cols`.
    cols : list[str] or None
        Cột cần chuẩn hoá. None → F0_COLS (12 cột gốc).
    baseline_n : int
    suffix : str
    rel_eps : float
        Sàn tương đối cho σ.
    report : bool
        In số nhóm (subject, trial, cột) bị kích hoạt sàn.

    Returns
    -------
    pd.DataFrame : df + các cột {col}{suffix}
    """
    cols = F0_COLS if cols is None else cols
    target = [c for c in cols if c in df.columns]
    if not target:
        raise ValueError("Không có cột nào trong `cols` tồn tại trong df")

    out = df.copy()
    n_floor = 0
    n_degenerate = 0

    for (subj, trial), grp in df.groupby(["subject", "trial"]):
        base = grp[grp["rep_idx"] <= baseline_n]
        mask = (out["subject"] == subj) & (out["trial"] == trial)
        for col in target:
            mu = base[col].mean()
            sg = base[col].std()
            if not np.isfinite(sg):
                sg = 0.0

            floor = rel_eps * abs(mu) if np.isfinite(mu) else 0.0
            if sg > floor:
                denom = sg
            elif floor > 0:
                denom = floor
                n_floor += 1
            else:
                # σ ≈ 0 và μ ≈ 0: baseline không có thông tin nào để neo vào
                out.loc[mask, f"{col}{suffix}"] = 0.0
                n_degenerate += 1
                continue

            out.loc[mask, f"{col}{suffix}"] = (out.loc[mask, col] - mu) / denom

    if report and (n_floor or n_degenerate):
        print(
            f"[apply_N1_zscore] sàn tương đối kích hoạt {n_floor} lần, "
            f"{n_degenerate} nhóm suy biến hoàn toàn (đặt z=0)"
        )
    return out


def apply_N1_ratio(df, baseline_n=3, cols=None):
    """
    N1 ratio normalization: r_k = x_k / x̄_{1:3}

    Chia mỗi feature cho mean của 3 rep đầu tiên trong mỗi (subject, trial).
    Ý nghĩa: giá trị ~1.0 = bình thường, <1.0 = suy giảm, >1.0 = tăng.
    Phù hợp cho amplitude features (TP, RMS) hơn z-score (không giả định
    phân phối chuẩn, dễ interpret hơn).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame chứa các cột F0 + metadata (subject, trial, rep_idx)
    baseline_n : int
        Số rep đầu dùng làm baseline (default=3, theo thầy)
    cols : list[str] or None
        Cột cần chuẩn hoá. None → F0_COLS (giữ hành vi cũ cho caller hiện có).

    Returns
    -------
    pd.DataFrame
        DataFrame gốc + các cột mới *_ratio
    """
    cols = F0_COLS if cols is None else cols
    out = df.copy()
    for (subj, trial), grp in df.groupby(["subject", "trial"]):
        base = grp[grp["rep_idx"] <= baseline_n]
        mask = (out["subject"] == subj) & (out["trial"] == trial)
        for col in cols:
            mu = base[col].mean()
            # Tránh chia cho 0: nếu mean baseline = 0, giữ nguyên raw value
            if abs(mu) < 1e-12:
                out.loc[mask, f"{col}_ratio"] = 1.0
            else:
                out.loc[mask, f"{col}_ratio"] = out.loc[mask, col] / mu
    return out


def apply_N1_diff(df, cols=None):
    """
    N1 differential: Δx_k = x_k − x_{k-1}

    Vi phân theo rep trong mỗi (subject, trial).
    Ý nghĩa: tốc độ thay đổi feature giữa 2 rep liên tiếp.
    Rep đầu tiên của mỗi trial: Δx_1 = 0 (không có rep trước đó).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame chứa các cột F0 + metadata
    cols : list[str] or None
        Cột cần vi phân. None → F0_COLS (giữ hành vi cũ cho caller hiện có).

    Returns
    -------
    pd.DataFrame
        DataFrame gốc + các cột mới *_diff
    """
    cols = F0_COLS if cols is None else cols
    out = df.copy()
    # Sort trước để diff() đúng thứ tự rep
    out = out.sort_values(["subject", "trial", "rep_idx"]).reset_index(drop=True)
    for col in cols:
        out[f"{col}_diff"] = out.groupby(["subject", "trial"])[col].diff().fillna(0)
    return out


def apply_rolling_slope(df, feature_col="MNF_mean_N1", window=5):
    """
    Rolling linear regression slope trên feature chuẩn hóa.

    Đây là feature sinh lý hợp lệ (không leak target), đã được xác nhận
    đóng góp +3.4pp RMSE improvement trong kết quả 16.8% GO.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đã có feature_col
    feature_col : str
        Tên cột feature để tính slope (default: MNF_mean_N1)
    window : int
        Số rep cho rolling window (default=5)

    Returns
    -------
    pd.DataFrame
        DataFrame gốc + 1 cột {feature_col}_slope{window}
    """
    out = df.copy()
    out = out.sort_values(["subject", "trial", "rep_idx"]).reset_index(drop=True)

    def _slope_fn(w):
        if len(w) < 2:
            return 0.0
        return np.polyfit(range(len(w)), w, 1)[0]

    slope_col = f"{feature_col}_slope{window}"
    out[slope_col] = out.groupby(["subject", "trial"])[feature_col].transform(
        lambda x: x.rolling(window, min_periods=2).apply(_slope_fn, raw=False)
    )
    out[slope_col] = out[slope_col].fillna(0)
    return out


def build_all_feature_sets(
    df, baseline_n=3, slope_window=5, cols=None, create_missing_zscore=True
):
    """
    Xây dựng tất cả feature sets theo đặc tả thầy.

    Áp dụng tuần tự:
    1. Z-score (nếu còn thiếu)
    2. Ratio normalization
    3. Differential features
    4. Rolling slope

    Trả về dict các feature sets và DataFrame đã enriched.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame gốc (đã có F0 + F1_N1 z-score từ notebook)
    baseline_n : int
        Số rep baseline cho ratio normalization
    slope_window : int
        Window size cho rolling slope
    cols : list[str] or None
        Tập đặc trưng GỐC cần xử lý. None → F0_COLS (12 cột, hành vi cũ).

        Truyền F0_COLS + EXTRA_BASE_COLS để mọi đặc trưng mới cũng nhận đủ
        cả ba biến thể neo (zscore/ratio/diff). Nếu không làm vậy, 42 cột mới
        sẽ CHỈ có z-score trong khi 12 cột cũ có đủ ba biến thể — khi đó A4
        so sánh không công bằng và kết quả là confound, không phải phát hiện.
    create_missing_zscore : bool
        True → tự tạo cột {col}_N1 còn thiếu bằng `apply_N1_zscore`. Cần thiết
        cho các cột mới chưa từng được chuẩn hoá trong notebook.

    Returns
    -------
    feature_sets : dict
        Mapping tên feature set → list tên cột
    df_enriched : pd.DataFrame
        DataFrame đầy đủ với tất cả features mới
    """
    cols = F0_COLS if cols is None else cols
    zscore_cols = [f"{c}_N1" for c in cols]
    ratio_cols = [f"{c}_ratio" for c in cols]
    diff_cols = [f"{c}_diff" for c in cols]

    # Step 0: z-score — tạo nếu thiếu, nếu không thì báo lỗi như trước
    missing_zscore = [c for c in zscore_cols if c not in df.columns]
    if missing_zscore:
        if not create_missing_zscore:
            raise ValueError(
                f"Missing N1 z-score columns: {missing_zscore}. "
                "Chạy apply_N1_normalization() trong notebook trước."
            )
        need = [c for c in cols if f"{c}_N1" not in df.columns]
        df = apply_N1_zscore(df, cols=need, baseline_n=baseline_n)

    # Step 1: Ratio normalization
    df = apply_N1_ratio(df, baseline_n=baseline_n, cols=cols)

    # Step 2: Differential features
    df = apply_N1_diff(df, cols=cols)

    # Step 3: Rolling slope (trên z-score MNF_mean, feature sinh lý chính)
    slope_col = f"MNF_mean_N1_slope{slope_window}"
    if slope_col not in df.columns:
        df = apply_rolling_slope(df, feature_col="MNF_mean_N1", window=slope_window)

    # ── Feature set definitions ──────────────────────────────────────────
    feature_sets = {
        # Từng biến thể riêng lẻ (cho ablation A1)
        "F0": list(cols),
        "F1_ZSCORE": zscore_cols,
        "F1_RATIO": ratio_cols,
        "F1_DIFF": diff_cols,
        # Tổ hợp
        "F1_ZSCORE_SLOPE": zscore_cols + [slope_col],
        "F1_FULL": zscore_cols + ratio_cols + diff_cols,
        "F1_FULL_SLOPE": zscore_cols + ratio_cols + diff_cols + [slope_col],
        # Tổ hợp 2-feature (cho ablation A1 chi tiết hơn)
        "F1_ZSCORE_RATIO": zscore_cols + ratio_cols,
        "F1_ZSCORE_DIFF": zscore_cols + diff_cols,
        "F1_RATIO_DIFF": ratio_cols + diff_cols,
    }

    return feature_sets, df


# ── Convenience: load + enrich in one call ─────────────────────────────────
def load_and_enrich(csv_path, baseline_n=3, slope_window=5):
    """
    Load per_cycle_features.csv và thêm tất cả N1 variants.

    Parameters
    ----------
    csv_path : str
        Path tới per_cycle_features.csv

    Returns
    -------
    feature_sets : dict
    df : pd.DataFrame
    """
    df = pd.read_csv(csv_path)
    return build_all_feature_sets(df, baseline_n=baseline_n, slope_window=slope_window)


if __name__ == "__main__":
    # Quick smoke test
    import os

    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "per_cycle_features.csv"
    )
    if os.path.exists(csv_path):
        feature_sets, df = load_and_enrich(csv_path)
        print(f"DataFrame shape: {df.shape}")
        print(f"\nFeature sets available:")
        for name, cols in feature_sets.items():
            print(f"  {name:25s} -> {len(cols)} features")
        # Sanity check: no NaN in new columns
        new_cols = F1_RATIO_COLS + F1_DIFF_COLS
        n_nan = df[new_cols].isna().sum().sum()
        print(f"\nNaN count in new features: {n_nan}")

    else:
        print(f"CSV not found at {csv_path}")
