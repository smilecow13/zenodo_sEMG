"""
Sinh bảng tham chiếu bicep Zenodo cho lần đo thử trên người thật
================================================================
Chỉ ĐỌC các file đã đóng băng, ghi ra reference/. Không ghi đè gì ngoài đó.

Hai đại lượng ρ, HAI pipeline, không được đặt lẫn:
  (W) ρ theo cửa sổ: MDF/MNF trên cửa sổ 4 s bước 2 s, Spearman theo thời gian.
      Nguồn: dataset/sEMG_data/trial_summary_with_flags.csv (toàn phổ, không cắt dải).
  (R) ρ theo rep: MDF_mean/MNF_mean mỗi rep (trung bình 3 phần của rep, dải 20–450 Hz),
      Spearman theo rep_idx, trên mọi rep của trial.
      Nguồn: data/processed/per_cycle_features_v2b.csv.
Hai cột này đo cùng hiện tượng nhưng khác cửa sổ, khác dải, khác trục x; số của mình
chỉ được đặt cạnh cột cùng pipeline.

    python build_reference.py              # sinh bảng từ CSV
    python build_reference.py --check-raw  # thêm: tái lập (W) từ dữ liệu thô
                                                    # Zenodo trên 3 trial bicep, assert khớp

Output (reference/):
  zenodo_bicep_window_rho.csv   26 trial, cột (W) + is_outlier
  zenodo_bicep_per_rep_rho.csv  26 trial, cột (R) + N_total + thời lượng rep
  zenodo_bicep_per_rep_v2b.csv  mọi rep bicep của v2b, để vẽ chồng đường của mình
  zenodo_bicep_pred_rho.csv     24 trial, ρ(FCF đoán, rep_idx) của G12 LOSO in-fold — tham
                                chiếu cho model_direction.py
  zenodo_bicep_summary.txt      phân vị của các cột ρ và hình dạng set

(P) ρ chiều của mô hình: Spearman(FCF đoán, rep_idx) trên các hàng GT2 (rep_idx > 3) của
    trial bicep, dự đoán out-of-fold của G12 (v2b, GT2, 12 cơ, mRMR theo nhóm in-fold k=12,
    RF 200/10/42), mô hình "train12_scoreall". Nguồn: data/results/g12_rowlevel_predictions_v2b.csv
    (decompose_bicep_p2.py). 24 trial chứ không phải 26: lọc IQR đã bỏ 2 trial bicep.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = HERE
OUT = os.path.join(HERE, "reference")
TS_CSV = os.path.join(BASE, "data", "processed", "trial_summary_with_flags.csv")
V2B_CSV = os.path.join(BASE, "data", "processed", "per_cycle_features_v2b.csv")
RAW_DIR = os.environ.get("ZENODO_SEMG_DIR", os.path.join(BASE, "dataset", "sEMG_data"))

BICEP_TRIALS = {5: "R BICEPS BRACHII", 6: "L BICEPS BRACHII"}
# cột kênh trong CSV thô Zenodo: trial 5 → kênh 0, trial 6 → kênh 1 (PRIME_MOVER)
RAW_COLS = {5: (0, 1), 6: (2, 3)}  # (time, emg)
CHECK_TRIALS = [(1, 5), (6, 5), (3, 6)]  # dài, ngắn nhất (11 cửa sổ), bên trái

N_BICEP_TRIALS = 26


def build_window(ts):
    w = ts[ts["trial"].isin(BICEP_TRIALS)].copy()
    assert len(w) == N_BICEP_TRIALS, len(w)
    assert set(w["muscle"]) == set(BICEP_TRIALS.values())
    cols = ["subject", "trial", "muscle", "duration_s", "n_windows",
            "MDF_baseline", "MDF_final", "MDF_pct_change", "MDF_spearman_rho",
            "MNF_spearman_rho", "is_outlier"]
    return w[cols].sort_values(["trial", "subject"]).reset_index(drop=True)


def build_per_rep(v2b):
    b = v2b[v2b["trial"].isin(BICEP_TRIALS)].copy()
    rows = []
    for (s, t), g in b.groupby(["subject", "trial"]):
        g = g.sort_values("rep_idx")
        assert g["N_total"].nunique() == 1
        rows.append({
            "subject": s, "trial": t, "muscle": g["muscle"].iloc[0],
            "N_total": int(g["N_total"].iloc[0]),
            "n_rows": len(g),
            "rep_dur_median_s": float(g["rep_duration_s"].median()),
            "rep_dur_max_s": float(g["rep_duration_s"].max()),
            "set_span_s": float(g["rep_start_s"].iloc[-1] - g["rep_start_s"].iloc[0]),
            "MDF_rep_rho": float(spearmanr(g["rep_idx"], g["MDF_mean"])[0]),
            "MNF_rep_rho": float(spearmanr(g["rep_idx"], g["MNF_mean"])[0]),
        })
    r = pd.DataFrame(rows)
    assert len(r) == N_BICEP_TRIALS, len(r)
    return r, b


PRED_CSV = os.path.join(BASE, "data", "results", "g12_rowlevel_predictions_v2b.csv")
PRED_MODEL = "train12_scoreall"
N_PRED_TRIALS, N_PRED_ROWS = 24, 2275  # 2275 = N_ROWS_B trong decompose_bicep_p2.py


def build_pred(pred):
    p = pred[(pred["model"] == PRED_MODEL) & pred["muscle"].isin(BICEP_TRIALS.values())]
    assert len(p) == N_PRED_ROWS, len(p)
    assert (p["rep_idx"] > 3).all() and (p["subject"] == p["test_subject"]).all()
    rows = []
    for (s, t), g in p.groupby(["subject", "trial"]):
        rows.append({"subject": s, "trial": t, "n_rows": len(g),
                     "pred_rho": float(spearmanr(g["rep_idx"], g["y_pred"])[0])})
    r = pd.DataFrame(rows)
    assert len(r) == N_PRED_TRIALS, len(r)
    return r


def check_raw(window):
    from scipy import signal
    sys.path.insert(0, HERE)
    from zenodo_windowed_mdf import SAMPLE_FREQ, window_rho

    b, a = signal.butter(4, [20, 450], btype="bandpass", fs=SAMPLE_FREQ)
    for s, t in CHECK_TRIALS:
        path = os.path.join(RAW_DIR, f"subject_{s}", f"trial_{t}.csv")
        if not os.path.exists(path):
            print(f"  bỏ qua S{s}/T{t}: không có dữ liệu thô")
            continue
        raw = pd.read_csv(path).values
        tc, ec = RAW_COLS[t]
        y = signal.filtfilt(b, a, np.nan_to_num(raw[:, ec].astype(float), nan=0.0))
        got = window_rho(y, raw[:, tc].astype(float))
        ref = window[(window["subject"] == s) & (window["trial"] == t)].iloc[0]
        assert got["n_windows"] == ref["n_windows"], (s, t, got["n_windows"], ref["n_windows"])
        for k in ("MDF_spearman_rho", "MNF_spearman_rho", "MDF_baseline"):
            assert abs(got[k] - ref[k]) < 1e-9, (s, t, k, got[k], ref[k])
        print(f"  S{s}/T{t}: tái lập khớp (n={got['n_windows']}, ρ_MDF={got['MDF_spearman_rho']:.5f})")


def build_mains():
    """Mốc nhiễu điện lưới cho qc_recording.py [6]: CHÍNH hàm mains_share, trên trial bicep thô
    Zenodo đi qua cùng pipeline_filter (ở đây fs đã là 1259 nên không resample)."""
    from qc_recording import mains_share, pipeline_filter

    rows = []
    for s in range(1, 14):
        for t in BICEP_TRIALS:
            path = os.path.join(RAW_DIR, f"subject_{s}", f"trial_{t}.csv")
            if not os.path.exists(path):
                sys.exit(f"--mains cần dữ liệu thô Zenodo: thiếu {path}")
            x = pd.read_csv(path).values[:, RAW_COLS[t][1]]
            rows.append({"subject": s, "trial": t,
                         "mains_share": mains_share(pipeline_filter(x, 1259))})
    m = pd.DataFrame(rows)
    assert len(m) == N_BICEP_TRIALS
    m.to_csv(os.path.join(OUT, "zenodo_bicep_mains_share.csv"), index=False)
    print(f"Mốc điện lưới Zenodo bicep ({len(m)} trial): {pct(m.mains_share)}")


def pct(x):
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return "  ".join(f"p{p}={v:+.3f}" for p, v in zip((10, 25, 50, 75, 90), q))


def main():
    os.makedirs(OUT, exist_ok=True)
    window = build_window(pd.read_csv(TS_CSV))
    per_rep, rep_rows = build_per_rep(pd.read_csv(V2B_CSV))
    assert set(zip(window.subject, window.trial)) == set(zip(per_rep.subject, per_rep.trial))
    pred = build_pred(pd.read_csv(PRED_CSV))
    assert set(zip(pred.subject, pred.trial)) <= set(zip(window.subject, window.trial))

    if "--check-raw" in sys.argv:
        print("Tái lập ρ theo cửa sổ từ dữ liệu thô:")
        check_raw(window)
    if "--mains" in sys.argv:
        build_mains()

    window.to_csv(os.path.join(OUT, "zenodo_bicep_window_rho.csv"), index=False)
    per_rep.to_csv(os.path.join(OUT, "zenodo_bicep_per_rep_rho.csv"), index=False)
    rep_rows.to_csv(os.path.join(OUT, "zenodo_bicep_per_rep_v2b.csv"), index=False)
    pred.to_csv(os.path.join(OUT, "zenodo_bicep_pred_rho.csv"), index=False)

    lines = [
        f"Zenodo bicep, {N_BICEP_TRIALS} trial (13 subject × trial 5 R, trial 6 L)",
        "",
        "(W) ρ theo cửa sổ 4 s/2 s, toàn phổ — trial_summary_with_flags.csv",
        f"  MDF: {pct(window.MDF_spearman_rho)}",
        f"  MNF: {pct(window.MNF_spearman_rho)}",
        f"  ρ_MDF ≥ 0: {(window.MDF_spearman_rho >= 0).sum()}/{len(window)} trial;"
        f" is_outlier: {int(window.is_outlier.sum())}",
        f"  số cửa sổ: {pct(window.n_windows)}",
        "",
        "(R) ρ theo rep, MDF_mean ~ rep_idx — per_cycle_features_v2b.csv",
        f"  MDF: {pct(per_rep.MDF_rep_rho)}",
        f"  MNF: {pct(per_rep.MNF_rep_rho)}",
        "",
        f"(P) ρ(FCF đoán, rep_idx), G12 LOSO in-fold, {N_PRED_TRIALS} trial — g12_rowlevel_predictions_v2b.csv",
        f"  {pct(pred.pred_rho)}",
        "",
        "Hình dạng set (theo bộ cắt rep của Zenodo)",
        f"  N_total: {pct(per_rep.N_total)}",
        f"  rep trung vị (s): {pct(per_rep.rep_dur_median_s)}",
        f"  thời lượng set (s): {pct(window.duration_s)}",
    ]
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "zenodo_bicep_summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
