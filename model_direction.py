"""
Kiểm CHIỀU của mô hình G12 Zenodo trên một set tự thu — không phải phép chấm độ chính xác
=========================================================================================
Câu hỏi duy nhất: FCF mô hình đoán có tăng theo số rep không, và mức tăng đó nằm ở đâu so với
chính mô hình ấy trên các trial bicep Zenodo mà nó chưa từng thấy.

Script KHÔNG in RMSE. Một người, một set: RMSE từng trial bicep trên chính Zenodo đã trải
13.5%–29.5% (G12, v2b, GT2), nên một con số RMSE ở đây không phân biệt được mạch tốt, mô hình
tốt hay may. Thêm vào đó là lệch thiết bị, người, vị trí điện cực, vật nặng.

Mô hình: 13 mô hình fold của cấu hình headline Zenodo, dựng lại y hệt decompose_bicep_p2.py
  v2b → filter_outlier_trials(min_reps=10) → prepare (N1 z/ratio/diff neo 3 rep đầu, bỏ
  rep_idx ≤ 3) → mRMR theo nhóm IN-FOLD, 12 nhóm × 3 biến thể = 36 cột → RF(200, depth 10, seed 42)
  Mỗi mô hình train trên 12 subject, danh sách cột chọn trong fold đó. Người tự thu không nằm
  trong train của mô hình nào, nên cả 13 đều ở đúng tình huống đã sinh phân bố tham chiếu
  reference/zenodo_bicep_pred_rho.csv (dự đoán out-of-fold). Không có danh sách chọn toàn cục,
  nên không có chọn lọc ngoài vòng CV.

Đặc trưng của set tự thu: đủ 54 đại lượng v2b trên tín hiệu đã resample về 1259 Hz và lọc
filtfilt 20–450 Hz, cắt rep bằng segment_reps (tham số Zenodo):
  12 cột F0 (MNF/MDF/TP/RMS × max/min/mean trên 3 phần rep) — logic notebook cell 11
  42 cột bổ sung — extra_features.extract_extra_per_rep, ngưỡng ZC/SSC/WAMP từ 3 rep đầu (v2b)

Cách dùng:
    python model_direction.py --train       # dựng 13 mô hình, assert khớp dự đoán cũ, lưu
    python model_direction.py --self-test   # đặc trưng + dự đoán trên trial Zenodo thô khớp v2b
    python model_direction.py recordings/x.csv --fs 2000 --col emg \\
        --start 14.2 --end 262.8

Output: recordings/<tên file>_out/model_preds.csv, model_summary.txt, model_direction.png
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.stats import spearmanr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = HERE
PREP = os.path.join(HERE, "zenodo_code")
# zenodo_code trước preprocessing: n1_features/extra_features/rep_segmentation lấy bản sao,
# phần còn lại của pipeline huấn luyện lấy từ preprocessing. verify_copies() bảo đảm hai bên
# trùng byte, nên thứ tự này không đổi kết quả.
sys.path[:0] = [HERE, os.path.join(HERE, "zenodo_code"), PREP]

import verify_copies  # noqa: E402

if verify_copies.main() != 0:
    sys.exit("Bản sao trong zenodo_code/ lệch bản gốc — dừng.")

from analyze_recording import load_column, run_pipeline  # noqa: E402
from eval_protocols import TARGET_SPECS  # noqa: E402
from extra_features import compute_baseline_thresholds, extract_extra_per_rep  # noqa: E402
from grouped_mrmr_p2 import fold_orders, group_cols  # noqa: E402
from iqr_filter import filter_outlier_trials  # noqa: E402
from n1_features import F0_COLS  # noqa: E402
from protocol2_final import ALL_BASE, RF_KWARGS, TARGET, prepare  # noqa: E402
from zenodo_windowed_mdf import SAMPLE_FREQ  # noqa: E402

V2B_CSV = os.path.join(BASE, "data", "processed", "per_cycle_features_v2b.csv")
V1_CSV = os.path.join(BASE, "data", "processed", "per_cycle_features.csv")
PRED_CSV = os.path.join(BASE, "data", "results", "g12_rowlevel_predictions_v2b.csv")
REF_P = os.path.join(HERE, "reference", "zenodo_bicep_pred_rho.csv")
MODEL_DIR = os.path.join(HERE, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "g12_v2b_fold_models.joblib")
RAW_DIR = os.environ.get("ZENODO_SEMG_DIR", os.path.join(BASE, "dataset", "sEMG_data"))

K = 12
N_FOLDS = 13
N_ROWS_12 = 10318
BASELINE_N = 3
MIN_REPS = 10  # như filter_outlier_trials(min_reps=10): set ngắn hơn không có trong train
BICEPS = ["R BICEPS BRACHII", "L BICEPS BRACHII"]


# ═══════════════════════════════════════════════════════════════════════════
# Huấn luyện 13 mô hình fold
# ═══════════════════════════════════════════════════════════════════════════
def train_fold_models():
    from sklearn.ensemble import RandomForestRegressor
    import joblib

    t0 = time.time()
    df_raw = filter_outlier_trials(pd.read_csv(V2B_CSV), min_reps=MIN_REPS, verbose=False)
    _, real12 = prepare(df_raw, shuffled=False)
    assert len(real12) == N_ROWS_12, len(real12)
    orders = fold_orders(real12, list(ALL_BASE), verbose=False)[0]
    assert sorted(orders) == list(range(1, N_FOLDS + 1))

    bic_rows = df_raw[df_raw["muscle"].isin(BICEPS)].groupby(["subject", "trial"])["N_total"].first()
    ref = pd.read_csv(PRED_CSV)
    ref = ref[ref["model"] == "train12_scoreall"]
    spec = TARGET_SPECS[TARGET]
    models = {}
    for subj, order in orders.items():
        cols = [c for g in order[:K] for c in group_cols(g)]
        assert len(cols) == 3 * K
        tr, te = real12[real12["subject"] != subj], real12[real12["subject"] == subj]
        m = RandomForestRegressor(**RF_KWARGS).fit(tr[cols].values, tr[TARGET].values)
        y = np.clip(m.predict(te[cols].values), *spec["y_range"])
        r = ref[ref["test_subject"] == subj]
        got = te[["subject", "trial", "rep_idx"]].assign(y=y).merge(
            r[["subject", "trial", "rep_idx", "y_pred"]], on=["subject", "trial", "rep_idx"])
        assert len(got) == len(te) == len(r), (subj, len(got), len(te), len(r))
        assert np.allclose(got["y"], got["y_pred"], rtol=0, atol=1e-12), \
            f"fold S{subj}: dự đoán không khớp g12_rowlevel_predictions_v2b.csv"
        models[subj] = {"cols": cols, "groups": order[:K], "model": m}
        print(f"  fold S{subj:>2}: {len(te)} hàng test khớp dự đoán cũ   [{time.time() - t0:.0f}s]")

    meta = {"n_rows_train_total": len(real12),
            "bicep_N_total_min": int(bic_rows.min()), "bicep_N_total_max": int(bic_rows.max()),
            "rf_kwargs": RF_KWARGS, "k_groups": K, "source": os.path.relpath(V2B_CSV, BASE)}
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"models": models, "meta": meta}, MODEL_PATH, compress=3)
    print(f"\n13/13 fold khớp tuyệt đối dự đoán out-of-fold đã đóng băng.\n→ {MODEL_PATH}")


def load_models():
    import joblib
    if not os.path.exists(MODEL_PATH):
        sys.exit("Chưa có mô hình. Chạy trước: python model_direction.py --train")
    bundle = joblib.load(MODEL_PATH)
    assert sorted(bundle["models"]) == list(range(1, N_FOLDS + 1))
    return bundle


# ═══════════════════════════════════════════════════════════════════════════
# Đặc trưng v2b cho một set
# ═══════════════════════════════════════════════════════════════════════════
def _mnf(pp, pf):
    if len(pp) == 0 or np.sum(pp) == 0:
        return None
    return np.sum(pf * pp) / np.sum(pp)


def _mdf(pp, pf):
    c = np.cumsum(pp)
    return pf[np.where(c >= c[-1] / 2)[0][0]]


def f0_per_rep(emg, fs, segments):
    """12 cột F0 — logic extract_per_cycle_features, preprocessing/code.ipynb cell 11."""
    rows = []
    for k, (s, e) in enumerate(segments):
        seg = emg[s:e]
        if len(seg) < int(fs * 0.5):
            continue
        L = len(seg) // 3
        if L < 16:
            continue
        thirds = [seg[:L], seg[L:2 * L], seg[2 * L:]]
        feat = {}
        for fname in ["MNF", "MDF", "TP", "RMS"]:
            vals = []
            for part in thirds:
                if len(part) < 16:
                    continue
                if fname == "RMS":
                    vals.append(np.sqrt(np.mean(part ** 2)))
                else:
                    fft_v = fft(part - np.mean(part))
                    freqs = fftfreq(len(part), 1.0 / fs)
                    band = (freqs >= 20) & (freqs <= 450)
                    pf, pp = freqs[band], np.abs(fft_v[band]) ** 2
                    if len(pp) == 0:
                        continue
                    if fname == "TP":
                        vals.append(np.sum(pp))
                    elif fname == "MNF":
                        vals.append(_mnf(pp, pf))
                    else:
                        vals.append(_mdf(pp, pf))
            if vals:
                feat[f"{fname}_max"] = np.max(vals)
                feat[f"{fname}_min"] = np.min(vals)
                feat[f"{fname}_mean"] = np.mean(vals)
        if feat:
            feat["rep_idx"] = k + 1
            rows.append(feat)
    return pd.DataFrame(rows)


def rep_table(emg, t, segments, subject=0, trial=0):
    f0 = f0_per_rep(emg, SAMPLE_FREQ, segments)
    thr = compute_baseline_thresholds(emg, segments, baseline_n=BASELINE_N)
    ex = extract_extra_per_rep(emg, t, SAMPLE_FREQ, segments, subject=subject, trial=trial,
                               thresholds=thr)
    f0 = f0.assign(subject=subject, trial=trial)
    df = f0.merge(ex, on=["subject", "trial", "rep_idx"], how="left")
    assert len(df) == len(f0)
    df["N_total"] = len(segments)
    df["FCF"] = df["rep_idx"] / len(segments)  # prepare() cần cột này; KHÔNG dùng để chấm
    df["muscle"] = "BICEPS"
    return df


def predict_all(bundle, feats_gt2):
    spec = TARGET_SPECS[TARGET]
    out = {}
    for subj, b in bundle["models"].items():
        X = feats_gt2[b["cols"]]
        assert X.notna().all().all(), f"fold S{subj}: có NaN trong {X.columns[X.isna().any()].tolist()}"
        out[subj] = np.clip(b["model"].predict(X.values), *spec["y_range"])
    return pd.DataFrame(out, index=feats_gt2["rep_idx"].values)


# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════
def self_test():
    """Nguồn đúng cho từng nhóm cột (đã chẩn đoán, không nới dung sai):
    - 12 cột F0 do notebook tính rồi ghi vào v1. build_features_v2.py đọc v1 bằng parser
      MẶC ĐỊNH rồi ghi ra v2, nên F0 trong v2/v2b là giá trị v1 đã lệch parser (tới ~1e-12
      tương đối ở TP/RMS). Trích lại → so với v1 đọc round_trip.
    - 42 cột bổ sung do build_features_v2/v2b tính rồi ghi thẳng → so với v2b đọc round_trip.
    - Chuỗi v1 --(parser mặc định)--> v2b cũng được assert, để chẩn đoán này không bị quên."""
    bundle = load_models()
    v1 = pd.read_csv(V1_CSV, float_precision="round_trip")
    v1d = pd.read_csv(V1_CSV)
    v2b = pd.read_csv(V2B_CSV, float_precision="round_trip")
    extra_cols = [c for c in ALL_BASE if c not in F0_COLS]
    assert len(extra_cols) == 42
    ref = pd.read_csv(PRED_CSV)
    ref = ref[ref["model"] == "train12_scoreall"]
    for s, tr, (tc, ec) in [(1, 5, (0, 1)), (3, 6, (2, 3))]:
        raw = pd.read_csv(os.path.join(RAW_DIR, f"subject_{s}", f"trial_{tr}.csv")).values
        res = run_pipeline(raw[:, ec], SAMPLE_FREQ, t_given=raw[:, tc].astype(float))
        feats = rep_table(res["emg"], res["t"], res["segments"], subject=s, trial=tr)
        sel = lambda d: d[(d.subject == s) & (d.trial == tr)].sort_values("rep_idx")  # noqa: E731
        w1, w1d, w2b = sel(v1), sel(v1d), sel(v2b)
        assert list(feats.rep_idx) == list(w1.rep_idx) == list(w2b.rep_idx)
        bad = [c for c in F0_COLS if not np.array_equal(feats[c].values, w1[c].values)]
        bad += [c for c in extra_cols if not np.array_equal(feats[c].values, w2b[c].values)]
        assert not bad, f"S{s}/T{tr}: {len(bad)} cột lệch nguồn, ví dụ {bad[:5]}"
        assert all(np.array_equal(w1d[c].values, w2b[c].values) for c in F0_COLS), \
            "chuỗi v1 → (parser mặc định) → v2b không còn đúng"
        print(f"S{s}/T{tr}: 12 cột F0 bit-identical với v1, 42 cột bổ sung bit-identical với v2b "
              f"({len(feats)} rep)")

        _, gt2 = prepare(feats, shuffled=False)
        pred = predict_all(bundle, gt2)
        r = ref[(ref.subject == s) & (ref.trial == tr)].set_index("rep_idx").sort_index()
        if len(r):  # trial còn sau lọc IQR: mô hình fold của chính subject này phải tái lập
            assert list(pred.index) == list(r.index)
            assert np.allclose(pred[s].values, r["y_pred"].values, rtol=0, atol=1e-9), \
                f"S{s}/T{tr}: dự đoán fold S{s} lệch dự đoán cũ"
            print(f"  fold S{s} đoán lại {len(r)} hàng GT2 khớp g12_rowlevel_predictions_v2b.csv")
    print("\nSelf-test PASS")


# ═══════════════════════════════════════════════════════════════════════════
# Một set tự thu
# ═══════════════════════════════════════════════════════════════════════════
def plot(pred, rhos, ref_rho, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    S1, REF = "#2a78d6", "#9a9893"
    fig, ax = plt.subplots(2, 1, figsize=(12, 7.5), facecolor=SURF,
                           gridspec_kw={"height_ratios": [1.6, 0.6]})
    for a in ax:
        a.set_facecolor(SURF)
        a.grid(True, color=GRID, lw=0.6)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color(INK2)
        a.tick_params(colors=INK2, labelsize=9)

    k = pred.index.values
    for c in pred.columns:
        ax[0].plot(k, pred[c].values, color=REF, lw=0.8, alpha=0.6)
    ax[0].plot(k, pred.median(axis=1).values, color=S1, lw=2.2, label="trung vị 13 mô hình")
    ax[0].plot([], [], color=REF, lw=0.8, label="từng mô hình fold")
    ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("rep_idx (bộ cắt rep)", color=INK2)
    ax[0].set_ylabel("FCF đoán", color=INK2)
    ax[0].set_title(f"FCF đoán theo rep — ρ trung vị 13 mô hình = {np.median(rhos):+.3f} "
                    f"[{min(rhos):+.2f}, {max(rhos):+.2f}]", color=INK, fontsize=11, loc="left")
    ax[0].legend(loc="upper left", fontsize=9, frameon=False)

    ax[1].scatter(ref_rho, np.zeros_like(ref_rho), s=60, color=REF, edgecolor=SURF, lw=1.5,
                  label=f"{len(ref_rho)} trial bicep Zenodo (out-of-fold)", zorder=2)
    ax[1].scatter([np.median(rhos)], [0], s=140, marker="D", color=S1, edgecolor=SURF, lw=2,
                  label="bản ghi này (trung vị)", zorder=3)
    ax[1].axvline(0, color=INK2, lw=0.8)
    ax[1].set_xlim(-1, 1)
    ax[1].set_yticks([])
    ax[1].set_xlabel("ρ(FCF đoán, rep_idx)", color=INK2)
    ax[1].legend(loc="upper left", fontsize=9, frameon=False, ncol=2)

    fig.suptitle(f"{title} — kiểm chiều, không phải phép chấm độ chính xác", color=INK,
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURF)
    plt.close(fig)


def run_recording(a):
    bundle = load_models()
    meta = bundle["meta"]
    x, col = load_column(a.path, a.col, a.time_col, a.fs)
    i0 = 0 if a.start is None else int(round(a.start * a.fs))
    i1 = len(x) if a.end is None else int(round(a.end * a.fs))
    assert 0 <= i0 < i1 <= len(x)
    if a.start is None or a.end is None:
        print("CHÚ Ý: chưa cắt theo video (--start/--end). Đoạn nghỉ đầu/cuối làm lệch ngưỡng cắt rep.")
    res = run_pipeline(x[i0:i1], a.fs, i0=i0)
    n_seg = len(res["segments"])
    if n_seg < MIN_REPS:
        sys.exit(f"Bộ cắt chỉ tìm được {n_seg} rep (< {MIN_REPS}); set ngắn như vậy không có trong train.")

    feats = rep_table(res["emg"], res["t"], res["segments"])
    _, gt2 = prepare(feats, shuffled=False)
    pred = predict_all(bundle, gt2)
    # assert cấu trúc TRƯỚC khi in
    assert pred.shape == (len(gt2), N_FOLDS), pred.shape
    assert (pred.index > BASELINE_N).all() and pred.index.is_monotonic_increasing
    assert len(gt2) >= 3
    rhos = [float(spearmanr(pred.index, pred[c])[0]) for c in pred.columns]
    ref = pd.read_csv(REF_P)
    assert len(ref) == 24
    med = float(np.median(rhos))

    lines = [
        "(P) ρ(FCF đoán, rep_idx) — KIỂM CHIỀU, không phải phép chấm độ chính xác",
        f"  số rep bộ cắt / hàng GT2 (rep_idx > 3): {n_seg} / {len(gt2)}",
        f"  ρ trung vị 13 mô hình fold : {med:+.3f}   [min {min(rhos):+.3f}, max {max(rhos):+.3f}]",
        f"  ρ > 0 ở                    : {sum(r > 0 for r in rhos)}/13 mô hình",
        f"  so với Zenodo bicep        : {int((ref.pred_rho > med).sum())}/24 trial có ρ cao hơn "
        f"(p10–p90: {np.percentile(ref.pred_rho, 10):+.2f} … {np.percentile(ref.pred_rho, 90):+.2f})",
        f"  N bộ cắt so với dải N_total bicep trong train [{meta['bicep_N_total_min']}, "
        f"{meta['bicep_N_total_max']}]: "
        + ("trong dải" if meta["bicep_N_total_min"] <= n_seg <= meta["bicep_N_total_max"]
           else "NGOÀI DẢI — mô hình chưa từng thấy set dài cỡ này"),
    ]
    text = "\n".join(lines)
    stem = os.path.splitext(os.path.basename(a.path))[0]
    od = os.path.join(os.path.dirname(os.path.abspath(a.path)), f"{stem}_out")
    os.makedirs(od, exist_ok=True)
    pred.rename(columns=lambda c: f"fold_S{c}").rename_axis("rep_idx").to_csv(
        os.path.join(od, "model_preds.csv"))
    head = f"file: {a.path}\ncột: {col}   fs gốc: {a.fs:g} SPS → {SAMPLE_FREQ} Hz\ncắt: {a.start} … {a.end} s\n"
    with open(os.path.join(od, "model_summary.txt"), "w", encoding="utf-8") as f:
        f.write(head + "\n" + text + "\n")
    plot(pred, rhos, ref.pred_rho.values, os.path.join(od, "model_direction.png"), stem)
    print(head)
    print(text)
    print(f"\n→ {od}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?")
    ap.add_argument("--fs", type=float)
    ap.add_argument("--col")
    ap.add_argument("--time-col")
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.train:
        return train_fold_models()
    if a.self_test:
        return self_test()
    if not a.path or not a.fs:
        ap.error("cần đường dẫn file và --fs (hoặc --train / --self-test)")
    return run_recording(a)


if __name__ == "__main__":
    sys.exit(main())
