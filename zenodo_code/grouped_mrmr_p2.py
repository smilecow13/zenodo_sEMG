"""
mRMR theo NHÓM ĐẠI LƯỢNG NỀN, in-fold, giao thức 2 (việc 1, mục 11 handoff v6)
==============================================================================
Lõi 12 cột `_N1` chọn in-fold dưới giao thức 2 cho RMSE 26.48%, trong khi
F1_FULL (162 cột) cho 24.39% — giá rút gọn +2.09pp, mất hơn 60% ưu thế so với
naive. Phép chọn đó có hai hạn chế:
  (a) chỉ quét 54 cột `_N1`, bỏ qua ratio/diff dù F1_FULL > F1_ZSCORE CONFIRMED;
  (b) đếm chi phí theo CỘT, trong khi chi phí trên ESP32 tỉ lệ với số ĐẠI LƯỢNG
      NỀN: có `ZC_mean` rồi thì `_N1`, `_ratio`, `_diff` chỉ là vài phép cộng
      trừ chia trên ba số baseline lưu sẵn.

THIẾT KẾ — KHAI BÁO TRƯỚC KHI CHẠY
  Dữ liệu    : per_cycle_features_v2.csv, filter_outlier_trials, 12 cơ, giao
               thức 2 (bỏ rep_idx ≤ 3 khỏi cả train lẫn test). 12 cơ là mẫu số
               chính theo DECISION_NOTES_go_denominator.md.
  Nhóm       : 54 đại lượng nền (18 tên × max/min/mean); nhóm g = {g_N1,
               g_ratio, g_diff}. Chọn g là lấy cả ba cột.
  Liên quan  : rel(g) = max qua 3 biến thể của |trung vị Spearman per-trial|,
               cùng thống kê với `rank_by_per_trial_rho` (min 10 rep, ≥3 giá
               trị phân biệt). Chỉ tính trên các trial của subject TRAIN.
  Dư thừa    : red(a,b) = trung bình |Pearson| của 3 cặp biến thể cùng loại
               (N1–N1, ratio–ratio, diff–diff), gộp hàng train.
  Tham lam   : điểm = rel − trung bình red với các nhóm đã chọn (trọng số 1.0,
               như `select_mrmr`). Thứ tự tham lam lồng nhau nên k nhỏ là tiền
               tố của k lớn: chạy tới K_MAX rồi cắt.
  Lưới k     : {6, 8, 12, 16, 20} đại lượng nền.
  Mô hình    : RF(n_estimators=200, max_depth=10, random_state=42), LOSO 13 fold.

  Ghi chú: trong một trial, z-score và ratio (μ ≠ 0) là biến đổi đơn điệu nên
  Spearman per-trial của `_N1`, `_ratio` bằng của cột thô. Thước đo liên quan
  vì thế KHÔNG thấy giá trị xuyên-subject của phép neo — đó là giới hạn đã biết
  và giữ nguyên để phép so với lõi-12 cũ chỉ đổi đúng một thứ: nhóm + biến thể.

HÀNG THAM CHIẾU (cùng tập hàng): naive, F0 (54 cột thô), F1_FULL (162 cột),
  CORE12_N1 (top-12 `_N1` bằng `run_loso_in_fold_selection` — phải tái lập
  26.48%).

HỌ H-GMRMR — Holm step-down trong họ, m = 4
  H1  CORE12_N1 vs G12      kỳ vọng DƯƠNG (nhóm hơn lõi cột lẻ, cùng số 12)
  H2  CORE12_N1 vs G20      kỳ vọng DƯƠNG
  H3  G12       vs F1_FULL  kỳ vọng DƯƠNG (rút gọn về 12 vẫn còn giá)
  H4  G20       vs F1_FULL  kỳ vọng DƯƠNG (rút gọn về 20 vẫn còn giá)
  H3/H4 INCONCLUSIVE KHÔNG có nghĩa là "không mất gì" — Wilcoxon không kiểm
  tương đương; báo kèm khoảng bootstrap của hiệu cặp theo fold.

MÔ TẢ (không kiểm định): đường giá theo k, GO của từng Gk so với F0, % ưu thế
  của F1_FULL so với naive được giữ lại, số đại lượng nền / số cột / số đại
  lượng tín hiệu phân biệt (ZC_mean và ZC_max chung một phép tính ZC), độ ổn
  định nhóm qua 13 fold, và cặp thật-vs-control cho F1_FULL, G12, G20.

Output (file mới, không ghi đè artifact cũ):
  data/results/grouped_mrmr_p2.csv            bảng tổng hợp
  data/results/grouped_mrmr_p2_folds.csv      RMSE từng fold từng cấu hình
  data/results/grouped_mrmr_p2_stability.csv  độ ổn định nhóm qua fold
  data/results/grouped_mrmr_p2_family.csv     họ H-GMRMR sau Holm
  data/results/grouped_mrmr_p2_cost_ci.csv    giá rút gọn + bootstrap CI

Usage:
    python grouped_mrmr_p2.py
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_protocols import TARGET_SPECS, _fit_predict, _score, compare_paired, run_loso
from feature_selection import (MIN_REPS_PER_TRIAL, rank_by_per_trial_rho,
                               run_loso_in_fold_selection)
from iqr_filter import filter_outlier_trials
from protocol2_final import ALL_BASE, RF_KWARGS, TARGET, V2_CSV, naive_loso, prepare
from stats_utils import bootstrap_ci
from verdict import EXPECT_POSITIVE, verdict

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "data", "results")

VARIANTS = ["_N1", "_ratio", "_diff"]
K_GRID = [6, 8, 12, 16, 20]
K_MAX = max(K_GRID)
ALPHA = 0.05


def group_cols(g):
    return [f"{g}{v}" for v in VARIANTS]


def signal_name(g):
    return g.rsplit("_", 1)[0]


# ═══════════════════════════════════════════════════════════════════════════
# Liên quan per-trial: mỗi trial chỉ dùng dữ liệu của chính nó, nên tính một
# lần cho mọi trial rồi lấy trung vị trên các trial TRAIN của từng fold là
# tương đương tuyệt đối với tính lại trong fold.
# ═══════════════════════════════════════════════════════════════════════════
def per_trial_rho_matrix(df, cols):
    rows, keys = [], []
    for key, grp in df.groupby(["subject", "trial"]):
        r = {}
        for col in cols:
            x = grp[col]
            if x.notna().sum() < MIN_REPS_PER_TRIAL or x.nunique() < 3:
                continue
            r[col] = x.corr(grp[TARGET], method="spearman")
        rows.append(r)
        keys.append(key)
    mat = pd.DataFrame(rows, columns=cols)
    mat.index = pd.MultiIndex.from_tuples(keys, names=["subject", "trial"])
    return mat


def fold_relevance(rho_mat, train_subjects):
    sub = rho_mat[rho_mat.index.get_level_values("subject").isin(train_subjects)]
    return sub.median(skipna=True).abs()


def grouped_mrmr_order(train, rho_mat, groups, k_max=K_MAX):
    rel_col = fold_relevance(rho_mat, train["subject"].unique())
    rel = pd.Series({g: np.nanmax([rel_col.get(c, np.nan) for c in group_cols(g)])
                     for g in groups})
    rel = rel[np.isfinite(rel)].sort_values(ascending=False)
    avail = list(rel.index)

    corr = {v: train[[f"{g}{v}" for g in avail]].corr().abs().to_numpy()
            for v in VARIANTS}
    red = np.nanmean(np.stack([corr[v] for v in VARIANTS]), axis=0)
    red = np.nan_to_num(red, nan=1.0)
    pos = {g: i for i, g in enumerate(avail)}

    selected = [avail[0]]
    while len(selected) < min(k_max, len(avail)):
        sel_idx = [pos[s] for s in selected]
        best, best_score = None, -np.inf
        for g in avail:
            if g in selected:
                continue
            score = rel[g] - red[pos[g], sel_idx].mean()
            if score > best_score:
                best, best_score = g, score
        selected.append(best)
    return selected, rel


def run_grouped(df, orders, k):
    spec = TARGET_SPECS[TARGET]
    out = []
    for subj, order in orders.items():
        cols = [c for g in order[:k] for c in group_cols(g)]
        tr, te = df[df["subject"] != subj], df[df["subject"] == subj]
        y_pred = _fit_predict(RandomForestRegressor, RF_KWARGS,
                              tr[cols].values, tr[TARGET].values, te[cols].values,
                              y_range=spec["y_range"])
        s = _score(te[TARGET].values, y_pred, scale_pct=spec["scale_pct"])
        s["test_subject"] = subj
        out.append(s)
    return pd.DataFrame(out)


def fold_orders(df, groups, verbose=True):
    rho_mat = per_trial_rho_matrix(df, [c for g in groups for c in group_cols(g)])
    orders = {}
    for subj in sorted(df["subject"].unique()):
        tr = df[df["subject"] != subj]
        orders[subj], _ = grouped_mrmr_order(tr, rho_mat, groups)
        if verbose:
            print(f"    fold S{subj}: {orders[subj][:6]} ...")
    return orders, rho_mat


def paired_delta_ci(res_a, res_b):
    m = res_a.merge(res_b, on="test_subject", suffixes=("_a", "_b"))
    return bootstrap_ci((m["rmse_a"] - m["rmse_b"]).values)


def main():
    t0 = time.time()
    print("=" * 84)
    print("mRMR THEO NHÓM ĐẠI LƯỢNG NỀN — in-fold, giao thức 2, 12 cơ")
    print("=" * 84)

    df_raw = filter_outlier_trials(pd.read_csv(V2_CSV), min_reps=10, verbose=False)
    sets_r, real = prepare(df_raw, shuffled=False)
    sets_c, ctrl = prepare(df_raw, shuffled=True)
    groups = list(ALL_BASE)
    assert len(groups) == 54 and all(c in real.columns
                                     for g in groups for c in group_cols(g))

    naive = naive_loso(real)["rmse"].mean()
    naive_c = naive_loso(ctrl)["rmse"].mean()
    print(f"\nn = {len(real)} reps (giao thức 2) | naive = {naive:.2f}% | "
          f"naive control = {naive_c:.2f}%")

    # ── Kiểm tra tương đương: relevance nhanh == rank_by_per_trial_rho ───────
    rho_chk = per_trial_rho_matrix(real, [f"{g}_N1" for g in groups])
    s0 = sorted(real["subject"].unique())[0]
    tr0 = real[real["subject"] != s0]
    fast = fold_relevance(rho_chk, tr0["subject"].unique()).sort_values(ascending=False)
    slow = rank_by_per_trial_rho(tr0, [f"{g}_N1" for g in groups], target=TARGET)
    diff = (fast.reindex(slow.index) - slow).abs().max()
    print(f"Kiểm tra relevance nhanh vs rank_by_per_trial_rho (fold S{s0}): "
          f"sai lệch max = {diff:.2e}")
    assert diff < 1e-9, "relevance nhanh không khớp hàm chuẩn"

    # ── Tham chiếu ────────────────────────────────────────────────────────
    res, res_c = {}, {}
    print("\n--- Tham chiếu ---")
    for cfg in ["F0", "F1_FULL"]:
        res[cfg] = run_loso(RandomForestRegressor, sets_r[cfg], TARGET, real,
                            model_kwargs=RF_KWARGS)
        res_c[cfg] = run_loso(RandomForestRegressor, sets_c[cfg], TARGET, ctrl,
                              model_kwargs=RF_KWARGS)
        print(f"  {cfg:10s} {len(sets_r[cfg]):>4d} cột  thật {res[cfg]['rmse'].mean():6.2f}%"
              f"  control {res_c[cfg]['rmse'].mean():6.2f}%   [{time.time()-t0:.0f}s]")

    core, core_log = run_loso_in_fold_selection(
        RandomForestRegressor, [f"{g}_N1" for g in groups], TARGET, real, k=12,
        model_kwargs=RF_KWARGS)
    res["CORE12_N1"] = core[["test_subject", "rmse", "mae", "r2"]]
    print(f"  CORE12_N1   12 cột  thật {core['rmse'].mean():6.2f}%   "
          f"(handoff v6 ghi 26.48%)   [{time.time()-t0:.0f}s]")

    # ── Thứ tự nhóm theo fold ───────────────────────────────────────────────
    print("\n--- Thứ tự mRMR nhóm trong từng fold (thật) ---")
    orders, _ = fold_orders(real, groups)
    orders_c, _ = fold_orders(ctrl, groups, verbose=False)

    for k in K_GRID:
        res[f"G{k}"] = run_grouped(real, orders, k)
        if k in (12, 20):
            res_c[f"G{k}"] = run_grouped(ctrl, orders_c, k)
        print(f"  G{k:<3d} xong   [{time.time()-t0:.0f}s]")

    # ── Bảng tổng hợp ───────────────────────────────────────────────────────
    f0 = res["F0"]["rmse"].mean()
    full = res["F1_FULL"]["rmse"].mean()
    f0_c = res_c["F0"]["rmse"].mean()
    adv_full = naive - full

    def n_signal(k):
        return np.mean([len({signal_name(g) for g in o[:k]}) for o in orders.values()])

    rows = [{"config": "naive", "n_bases": 0, "n_cols": 0, "n_signals": 0,
             "rmse": naive, "rmse_control": naive_c}]
    spec_rows = [("F0", 54, 54, 18), ("F1_FULL", 54, 162, 18), ("CORE12_N1", 12, 12, None)]
    for cfg, nb, nc, ns in spec_rows:
        rows.append({"config": cfg, "n_bases": nb, "n_cols": nc, "n_signals": ns,
                     "rmse": res[cfg]["rmse"].mean(),
                     "rmse_control": res_c[cfg]["rmse"].mean() if cfg in res_c else np.nan})
    for k in K_GRID:
        rows.append({"config": f"G{k}", "n_bases": k, "n_cols": 3 * k,
                     "n_signals": n_signal(k), "rmse": res[f"G{k}"]["rmse"].mean(),
                     "rmse_control": res_c[f"G{k}"]["rmse"].mean()
                     if f"G{k}" in res_c else np.nan})
    tbl = pd.DataFrame(rows)
    tbl["GO_vs_F0_pct"] = (f0 - tbl["rmse"]) / f0 * 100
    tbl["vs_naive_pct"] = (naive - tbl["rmse"]) / naive * 100
    tbl["cost_vs_FULL_pp"] = tbl["rmse"] - full
    tbl["pct_FULL_advantage_kept"] = (naive - tbl["rmse"]) / adv_full * 100
    tbl["gain_real_pp"] = f0 - tbl["rmse"]
    tbl["gain_control_pp"] = f0_c - tbl["rmse_control"]

    print("\n" + "=" * 84)
    print("BẢNG (12 cơ, giao thức 2; mẫu số GO = F0 cùng bộ, cùng tập hàng)")
    print("=" * 84)
    print(f"{'cấu hình':11s}{'#nền':>6s}{'#cột':>6s}{'#tín hiệu':>10s}{'RMSE':>8s}"
          f"{'GO':>8s}{'>naive':>8s}{'giá':>8s}{'giữ %':>7s}{'lợi thật':>10s}{'lợi ctrl':>10s}")
    for _, r in tbl.iterrows():
        ns = "" if pd.isna(r.n_signals) else f"{r.n_signals:.1f}"
        gc = "" if pd.isna(r.gain_control_pp) else f"{r.gain_control_pp:+.2f}"
        print(f"{r.config:11s}{r.n_bases:>6d}{r.n_cols:>6d}{ns:>10s}{r.rmse:>7.2f}%"
              f"{r.GO_vs_F0_pct:>7.2f}%{r.vs_naive_pct:>7.2f}%{r.cost_vs_FULL_pp:>+8.2f}"
              f"{r.pct_FULL_advantage_kept:>7.1f}{r.gain_real_pp:>+10.2f}{gc:>10s}")

    print("\n  Giá rút gọn so với F1_FULL, bootstrap 95% theo fold (dương = kém hơn FULL):")
    ci_rows = []
    for cfg in ["CORE12_N1"] + [f"G{k}" for k in K_GRID]:
        ci = paired_delta_ci(res[cfg], res["F1_FULL"])
        ci_rows.append({"config": cfg, **ci})
        print(f"    {cfg:10s} {ci['point_estimate']:+.2f}pp  "
              f"[{ci['ci_lower']:+.2f}, {ci['ci_upper']:+.2f}]")

    # ── Họ H-GMRMR, Holm step-down ─────────────────────────────────────────
    print("\n" + "=" * 84)
    print("HỌ H-GMRMR (Holm step-down, m=4)")
    print("=" * 84)
    fam = [
        ("CORE12_N1", "G12", "nhóm hơn lõi cột lẻ (12 vs 12)"),
        ("CORE12_N1", "G20", "20 nhóm hơn lõi 12 cột lẻ"),
        ("G12", "F1_FULL", "rút gọn về 12 nhóm vẫn còn giá"),
        ("G20", "F1_FULL", "rút gọn về 20 nhóm vẫn còn giá"),
    ]
    cmps = [(a, b, lbl, compare_paired(res[a], res[b], a, b)) for a, b, lbl in fam]
    m = len(cmps)
    stopped = False
    fam_rows = []
    for rank, (a, b, lbl, cm) in enumerate(sorted(cmps, key=lambda x: x[3]["p_value"])):
        a_holm = ALPHA / (m - rank)
        if not stopped and not (cm["p_value"] < a_holm):
            stopped = True
        # step-down: sau lần đầu không bác bỏ, mọi kiểm định sau đều không bác bỏ
        v = verdict(cm["p_value"], cm["delta_mean"], EXPECT_POSITIVE,
                    alpha=0.0 if stopped else a_holm, claim=lbl)
        print(v)
        fam_rows.append({"a": a, "b": b, "claim": lbl, "delta_pp": cm["delta_mean"],
                         "p_value": cm["p_value"], "holm_alpha": a_holm,
                         "state": v.state})

    # ── Độ ổn định ──────────────────────────────────────────────────────────
    n_folds = len(orders)
    stab = []
    for g in groups:
        ranks = [o.index(g) + 1 if g in o else np.nan for o in orders.values()]
        stab.append({"base": g, "signal": signal_name(g),
                     "folds_in_G12": sum(1 for o in orders.values() if g in o[:12]),
                     "folds_in_G20": sum(1 for o in orders.values() if g in o[:20]),
                     "median_rank": np.nanmedian(ranks) if np.isfinite(ranks).any() else np.nan})
    stab = pd.DataFrame(stab).sort_values(["folds_in_G12", "folds_in_G20", "median_rank"],
                                          ascending=[False, False, True])
    print(f"\n--- Độ ổn định nhóm qua {n_folds} fold ---")
    print(stab[stab.folds_in_G20 > 0].head(26).to_string(index=False))
    for k in (12, 20):
        stable = stab[stab[f"folds_in_G{k}"] == n_folds]
        print(f"\n  Có mặt ở cả {n_folds} fold trong G{k}: {len(stable)} nhóm, "
              f"{stable.signal.nunique()} đại lượng tín hiệu -> {list(stable.base)}")

    # ── Ghi ─────────────────────────────────────────────────────────────────
    tbl.to_csv(os.path.join(RESULTS_DIR, "grouped_mrmr_p2.csv"), index=False)
    folds = pd.concat([r.assign(config=c, branch="real") for c, r in res.items()] +
                      [r.assign(config=c, branch="control") for c, r in res_c.items()])
    folds.to_csv(os.path.join(RESULTS_DIR, "grouped_mrmr_p2_folds.csv"), index=False)
    stab.to_csv(os.path.join(RESULTS_DIR, "grouped_mrmr_p2_stability.csv"), index=False)
    pd.DataFrame(fam_rows).to_csv(
        os.path.join(RESULTS_DIR, "grouped_mrmr_p2_family.csv"), index=False)
    pd.DataFrame(ci_rows).to_csv(
        os.path.join(RESULTS_DIR, "grouped_mrmr_p2_cost_ci.csv"), index=False)
    print(f"\nĐã ghi 5 file grouped_mrmr_p2*.csv vào {RESULTS_DIR}  "
          f"[tổng {time.time()-t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
