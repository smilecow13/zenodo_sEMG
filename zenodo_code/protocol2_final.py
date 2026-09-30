"""
Giao thức 2: GO chính thức + chọn lại lõi feature trên k>3
===========================================================
Giao thức 2 = KHÔNG chấm điểm trên 3 rep baseline. Ba rep đó dùng để tính μ, σ
cho neo N1, nên chấm điểm trên chính chúng là double-dipping, và nó mở một kênh
rò rỉ có dạng đại số chính xác (|z| ≤ 2/√3 đồng thời trên mọi cột ở rep ≤ 3;
đo được 100% rep baseline thoả so với 0.01% rep còn lại).

Rep baseline bị loại khỏi CẢ train LẪN test. Giữ chúng trong train thì vô hại
(luật "mọi |z| nhỏ → FCF thấp" sẽ không bao giờ kích hoạt trên test toàn k>3)
nhưng mô tả rắc rối hơn mà không được gì.

HAI VIỆC:
  1. GO CHÍNH THỨC dưới giao thức 2 — chạy thật thay vì ngoại suy theo tỷ lệ.
     Ước lượng theo tỷ lệ trước đó (~8–9%) chỉ là phép nhân 13.46% × 0.64; con
     số mang ra mốc go/no-go phải là con số chạy ra.
  2. CHỌN LẠI LÕI FEATURE trên các hàng k>3, in-fold. Lõi 11 feature cũ được
     chọn trên tập còn đủ rep baseline, nên cần chạy lại cho nhất quán với
     giao thức mới.

HỌ GIẢ THUYẾT — KHAI BÁO TRƯỚC KHI CHẠY (quy ước mục 4):
  H-P2-NORM  (3 kiểm định) chuẩn hoá neo có hơn đặc trưng thô dưới giao thức 2
               - F0 vs F1_ZSCORE        kỳ vọng dương
               - F0 vs F1_FULL          kỳ vọng dương
               - F1_ZSCORE vs F1_FULL   kỳ vọng dương
  H-P2-CTRL  (2 kiểm định) control xáo trộn có rò rỉ không
               - F0ctrl vs F1_ZSCOREctrl  kỳ vọng dương (nếu rò rỉ)
               - F0ctrl vs F1_FULLctrl    kỳ vọng dương (nếu rò rỉ)
  Ngưỡng Holm trong họ: alpha/(m−rank), m = số kiểm định của họ đó.

Output: data/results/go_official_protocol2.csv
        data/results/ablation_main_with_control.csv
        data/results/selection_stability_core_k_gt3.csv

Usage:
    python protocol2_final.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_protocols import compare_paired, run_loso
from extra_features import EXTRA_BASE_COLS
from feature_selection import run_loso_in_fold_selection
from iqr_filter import filter_outlier_trials
from n1_boundary_check import shuffle_rows_within_trial
from n1_features import F0_COLS, apply_N1_zscore, build_all_feature_sets
from verdict import EXPECT_POSITIVE, from_compare

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_CSV = os.path.join(BASE, "data", "processed", "per_cycle_features_v2.csv")
RESULTS_DIR = os.path.join(BASE, "data", "results")

TARGET = "FCF"
RF_KWARGS = {"n_estimators": 200, "max_depth": 10, "n_jobs": -1, "random_state": 42}
ALL_BASE = F0_COLS + EXTRA_BASE_COLS
BASELINE_N = 3
SEED = 42
ALPHA = 0.05

# Cấu hình KHÔNG chứa slope (theo decision note: slope bị loại khỏi kết quả chính)
CONFIGS = ["F0", "F1_ZSCORE", "F1_FULL"]


def naive_loso(df, target=TARGET):
    rows = []
    for subj in sorted(df["subject"].unique()):
        tr, te = df[df["subject"] != subj], df[df["subject"] == subj]
        if len(te) < 3:
            continue
        err = te[target].values - tr[target].mean()
        rows.append({"test_subject": subj,
                     "rmse": float(np.sqrt(np.mean(err**2)) * 100)})
    return pd.DataFrame(rows)


def prepare(df_raw, shuffled):
    """Chuẩn bị dataframe giao thức 2 (đã bỏ rep baseline) + feature sets."""
    base = df_raw
    if shuffled:
        base = shuffle_rows_within_trial(base, ALL_BASE, seed=SEED)
    # tính neo N1 từ đầu để hai nhánh dùng đúng một công thức
    drop = [c for c in base.columns if c.endswith("_N1")]
    enriched = apply_N1_zscore(base.drop(columns=drop), cols=ALL_BASE,
                               baseline_n=BASELINE_N, report=False)
    sets, enriched = build_all_feature_sets(enriched, cols=ALL_BASE)
    # Giao thức 2: bỏ rep baseline khỏi cả train lẫn test (neo đã tính xong)
    return sets, enriched[enriched["rep_idx"] > BASELINE_N].reset_index(drop=True)


def main():
    print("=" * 78)
    print("GIAO THỨC 2: GO chính thức + chọn lại lõi feature (k>3)")
    print("=" * 78)

    df_raw = filter_outlier_trials(pd.read_csv(V2_CSV), min_reps=10, verbose=False)
    sets_real, real = prepare(df_raw, shuffled=False)
    sets_ctrl, ctrl = prepare(df_raw, shuffled=True)

    print(f"\nTrước khi bỏ rep baseline: {len(df_raw)} reps")
    print(f"Giao thức 2 (k>3)        : {len(real)} reps "
          f"(bỏ {len(df_raw) - len(real)} = {(len(df_raw)-len(real))/len(df_raw)*100:.2f}%)")

    naive = naive_loso(real)["rmse"].mean()
    naive_c = naive_loso(ctrl)["rmse"].mean()
    print(f"Naive (cùng tập hàng)    : {naive:.2f}%  | control: {naive_c:.2f}%")

    # ── VIỆC 1: GO chính thức ──────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("VIỆC 1: GO chính thức dưới giao thức 2")
    print("=" * 78)

    res_r, res_c, rows = {}, {}, []
    print(f"\n{'cấu hình':16s}{'#feat':>7s}{'THẬT':>9s}{'CONTROL':>10s}{'khoảng cách':>13s}")
    print("-" * 60)
    for cfg in CONFIGS:
        cr = [c for c in sets_real[cfg] if c in real.columns]
        cc = [c for c in sets_ctrl[cfg] if c in ctrl.columns]
        r = run_loso(RandomForestRegressor, cr, TARGET, real, model_kwargs=RF_KWARGS)
        c = run_loso(RandomForestRegressor, cc, TARGET, ctrl, model_kwargs=RF_KWARGS)
        res_r[cfg], res_c[cfg] = r, c
        rm, cm = r["rmse"].mean(), c["rmse"].mean()
        print(f"{cfg:16s}{len(cr):>7d}{rm:>8.2f}%{cm:>9.2f}%{cm - rm:>+12.2f}pp")
        rows.append({"config": cfg, "n_features": len(cr),
                     "rmse_real": rm, "rmse_control": cm,
                     "naive_real": naive, "naive_control": naive_c})

    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(RESULTS_DIR, "ablation_main_with_control.csv"), index=False)

    f0 = tbl.loc[tbl.config == "F0", "rmse_real"].iloc[0]
    best_cfg = tbl.loc[tbl.rmse_real.idxmin(), "config"]
    best = tbl.rmse_real.min()
    go = (f0 - best) / f0 * 100
    vs_naive = (naive - best) / naive * 100

    print(f"\n  GO CHÍNH THỨC (giao thức 2) = ({f0:.2f} − {best:.2f})/{f0:.2f} "
          f"= {go:.2f}%   [tốt nhất: {best_cfg}]")
    print(f"  Hơn naive = {vs_naive:.2f}%")
    print(f"  Ngưỡng 15%: {'ĐẠT' if go >= 15 else 'CHƯA ĐẠT'}")

    # Chuỗi ba lần chỉnh
    print("\n  Chuỗi chỉnh con số GO:")
    print("    12.78%  ban đầu (có slope, chấm điểm cả rep baseline)")
    print("    13.46%  bỏ cấu hình slope, thêm 42 đặc trưng mới")
    print(f"    {go:5.2f}%  giao thức 2 — bỏ double-dipping  <-- CHÍNH THỨC")

    # ── Kiểm định, dùng verdict() với chiều kỳ vọng khai báo rõ ────────────
    print("\n--- HỌ H-P2-NORM (Holm trong họ, m=3) ---")
    pairs_norm = [
        ("F0", "F1_ZSCORE", "chuẩn hoá z-score hơn đặc trưng thô"),
        ("F0", "F1_FULL", "bộ neo đầy đủ hơn đặc trưng thô"),
        ("F1_ZSCORE", "F1_FULL", "thêm ratio+diff hơn chỉ z-score"),
    ]
    cmps = [(lbl, compare_paired(res_r[a], res_r[b], f"{a}-real", f"{b}-real"))
            for a, b, lbl in pairs_norm]
    m = len(cmps)
    for rank, (lbl, cm) in enumerate(sorted(cmps, key=lambda x: x[1]["p_value"])):
        v = from_compare(cm, EXPECT_POSITIVE, alpha=ALPHA / (m - rank), claim=lbl)
        print(v)

    print("\n--- HỌ H-P2-CTRL (Holm trong họ, m=2) ---")
    pairs_ctrl = [
        ("F0", "F1_ZSCORE", "control: z-score có rò rỉ biên"),
        ("F0", "F1_FULL", "control: bộ neo đầy đủ có rò rỉ biên"),
    ]
    cmps_c = [(lbl, compare_paired(res_c[a], res_c[b], f"{a}-ctrl", f"{b}-ctrl"))
              for a, b, lbl in pairs_ctrl]
    m = len(cmps_c)
    for rank, (lbl, cm) in enumerate(sorted(cmps_c, key=lambda x: x[1]["p_value"])):
        v = from_compare(cm, EXPECT_POSITIVE, alpha=ALPHA / (m - rank), claim=lbl)
        print(v)
        if v.is_reversed:
            print("      -> ngược chiều = N1 còn KÉM hơn F0 trên dữ liệu xáo trộn,")
            print("         tức bằng chứng KHÔNG rò rỉ, không phải dấu hiệu có vấn đề.")

    print("\n  Cặp số nên đưa vào bảng chính (vững hơn p-value, không phụ thuộc họ):")
    for cfg in ["F1_ZSCORE", "F1_FULL"]:
        gr = f0 - tbl.loc[tbl.config == cfg, "rmse_real"].iloc[0]
        f0c = tbl.loc[tbl.config == "F0", "rmse_control"].iloc[0]
        gc = f0c - tbl.loc[tbl.config == cfg, "rmse_control"].iloc[0]
        print(f"    {cfg:12s} lợi THẬT {gr:+.2f}pp  vs  lợi CONTROL {gc:+.2f}pp")

    # ── VIỆC 2: chọn lại lõi trên k>3 ──────────────────────────────────────
    print("\n" + "=" * 78)
    print("VIỆC 2: chọn lại lõi feature in-fold trên k>3")
    print("=" * 78)

    cands = [f"{c}_N1" for c in ALL_BASE]
    res_sel, log = run_loso_in_fold_selection(
        RandomForestRegressor, cands, TARGET, real, k=12, model_kwargs=RF_KWARGS
    )
    n_folds = res_sel["test_subject"].nunique()
    print(f"\n  top-12 in-fold (k>3): RMSE = {res_sel['rmse'].mean():.2f}% "
          f"± {res_sel['rmse'].std():.2f}")
    print(f"\n  Độ ổn định qua {n_folds} fold:")
    print(log.head(16).to_string(index=False))
    log.to_csv(os.path.join(RESULTS_DIR, "selection_stability_core_k_gt3.csv"),
               index=False)

    new_core = sorted(log[log.n_folds_selected == n_folds]["feature"])
    OLD_CORE = sorted([
        "ZC_mean_N1", "ZC_max_N1", "ZC_min_N1",
        "SSC_mean_N1", "SSC_max_N1", "SSC_min_N1",
        "MNF_mean_N1", "MNF_max_N1", "MNF_min_N1",
        "MDF_mean_N1", "SpecEnt_mean_N1",
    ])
    print(f"\n  Lõi mới ({len(new_core)} feature ở mọi fold):")
    for f in new_core:
        print(f"    {f}{'' if f in OLD_CORE else '   <-- MỚI'}")
    gone = [f for f in OLD_CORE if f not in new_core]
    if gone:
        print(f"  Rời khỏi lõi: {gone}")
    else:
        print("  Không feature nào rời khỏi lõi cũ.")

    pd.DataFrame([{
        "protocol": "2 (k>3)", "n_reps": len(real), "naive": naive,
        "f0": f0, "best_config": best_cfg, "best_rmse": best,
        "GO_pct": go, "vs_naive_pct": vs_naive,
        "core_size": len(new_core), "core": ";".join(new_core),
    }]).to_csv(os.path.join(RESULTS_DIR, "go_official_protocol2.csv"), index=False)

    print(f"\nĐã ghi 3 file vào {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
