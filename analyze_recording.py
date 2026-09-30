"""
Phân tích một set bicep curl tự thu bằng đúng pipeline Zenodo
=============================================================
Đầu vào: một file CSV/TXT trong recordings/, có một cột sEMG (một kênh), lấy mẫu
đều ở tần số --fs. Đơn vị biên độ không quan trọng, vì ρ của MDF không đổi theo thang.

Chuỗi xử lý (giống Zenodo, trừ bước 2):
  1. cắt bản ghi về [--start, --end] giây, lấy mốc từ video (rep đầu → rep cuối)
  2. resample về 1259 Hz bằng resample_poly (có lọc chống aliasing), dựng lại t = n/1259
  3. Butterworth bậc 4, 20–450 Hz, filtfilt (không nhân quả, như Zenodo)
  4. (W) MDF/MNF theo cửa sổ 4 s, bước 2 s, toàn phổ → ρ Spearman theo thời gian
  5. (R) cắt rep bằng rep_segmentation.segment_reps (tham số mặc định Zenodo), MDF/MNF mỗi rep
     = trung bình 3 phần, dải 20–450 Hz → ρ Spearman theo rep_idx

Script KHÔNG in phán quyết "thấy / không thấy". Nó in các thành phần (dấu của ρ, |ρ| so với
ngưỡng tới hạn ứng với đúng số cửa sổ), để đọc theo quy tắc đã khai trước trong README.

Cách dùng:
    python analyze_recording.py --self-test
    python analyze_recording.py <file> --fs 2000 --col emg --start 14.2 --end 262.8 --reps-only
    python analyze_recording.py recordings/buoi1_set1.csv --fs 2000 \\
        --col emg --start 12.4 --end 281.0 --n-video 88

Output: recordings/<tên file>_out/ (git bỏ qua cùng recordings/)
  windows.csv   mỗi cửa sổ: t_center, MDF, MNF
  rep_boundaries.csv  PHÂN ĐOẠN: mọi đoạn giữa hai biên (rep / dropped), thời gian theo đồng
                      hồ bản ghi gốc, chỉ số mẫu trong file gốc — cột theo wsd_rep_boundaries.csv
  reps.csv      mỗi rep: rep_idx, start/end, thời lượng, MDF_mean, MNF_mean
  dropped.csv   các đoạn giữa hai valley bị bộ cắt bỏ vì ngoài [1.2, 4.5] s
  summary.txt   mọi con số in ra màn hình
  diagnostic.png
"""

import argparse
import os
import sys
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import spearmanr, t as t_dist

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = HERE
sys.path.insert(0, os.path.join(HERE, "zenodo_code"))
sys.path.insert(0, HERE)

from rep_segmentation import segment_reps  # noqa: E402
from zenodo_windowed_mdf import SAMPLE_FREQ, STEP_S, WINDOW_S, calculate_median_frequencies  # noqa: E402

REF_W = os.path.join(HERE, "reference", "zenodo_bicep_window_rho.csv")
REF_REPS = os.path.join(HERE, "reference", "zenodo_bicep_per_rep_v2b.csv")
RAW_DIR = os.environ.get("ZENODO_SEMG_DIR", os.path.join(BASE, "dataset", "sEMG_data"))
MIN_REP_S, MAX_REP_S = 1.2, 4.5  # mặc định của segment_reps, chỉ dùng để báo đoạn bị bỏ


# ── per-rep: chép logic MNF/MDF của extract_per_cycle_features (preprocessing/code.ipynb,
#    cell 11), nguồn của cột MDF_mean/MNF_mean trong v1 → v2b. Self-test assert khớp v2b.
def _mnf(pp, pf):
    return np.sum(pf * pp) / np.sum(pp)


def _mdf(pp, pf):
    c = np.cumsum(pp)
    return pf[np.where(c >= c[-1] / 2)[0][0]]


def per_rep_features(emg, t, fs, segments):
    rows = []
    for k, (s, e) in enumerate(segments):
        seg, tseg = emg[s:e], t[s:e]
        if len(seg) < int(fs * 0.5):
            continue
        L = len(seg) // 3
        if L < 16:
            continue
        thirds = [seg[:L], seg[L:2 * L], seg[2 * L:]]
        mnf, mdf = [], []
        for part in thirds:
            f_v = fft(part - np.mean(part))
            freqs = fftfreq(len(part), 1.0 / fs)
            band = (freqs >= 20) & (freqs <= 450)
            pf, pp = freqs[band], np.abs(f_v[band]) ** 2
            if len(pp) == 0:
                continue
            mnf.append(_mnf(pp, pf))
            mdf.append(_mdf(pp, pf))
        if not mdf:
            continue
        rows.append({"rep_idx": k + 1, "start_s": float(tseg[0]), "end_s": float(tseg[-1]),
                     "rep_duration_s": float(tseg[-1] - tseg[0]),
                     "MDF_mean": float(np.mean(mdf)), "MNF_mean": float(np.mean(mnf))})
    return pd.DataFrame(rows)


def critical_rho(n, alpha=0.05):
    """|ρ| tới hạn hai phía dưới H0 không xu hướng (xấp xỉ t, n−2 bậc tự do)."""
    tc = t_dist.ppf(1 - alpha / 2, n - 2)
    return float(tc / np.sqrt(n - 2 + tc ** 2))


def to_zenodo_rate(x, fs):
    if fs == SAMPLE_FREQ:
        return x
    fr = Fraction(SAMPLE_FREQ, int(round(fs))).limit_denominator(10000)
    assert abs(fs - round(fs)) < 1e-9, "--fs phải là số nguyên (SPS)"
    return signal.resample_poly(x, fr.numerator, fr.denominator)


PIPELINE_TAG = "zenodo_segment_reps / resample→1259 Hz, filtfilt 20–450 Hz"


def run_pipeline(x, fs, t_given=None, i0=0):
    """x: sEMG thô đã cắt về khoảng set, bắt đầu ở mẫu i0 của file gốc. Mọi thời gian trả về
    tính theo ĐỒNG HỒ CỦA BẢN GHI GỐC (t = i0/fs + n/1259), để đối chiếu thẳng với video.
    t_given chỉ dùng khi fs==1259 (self-test trên trục thời gian gốc của Zenodo)."""
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0)
    y = to_zenodo_rate(x, fs)
    if t_given is not None and fs == SAMPLE_FREQ:
        t = t_given
    else:
        t = i0 / fs + np.arange(len(y)) / SAMPLE_FREQ
    assert len(t) == len(y)
    b, a = signal.butter(4, [20, 450], btype="bandpass", fs=SAMPLE_FREQ)
    emg = signal.filtfilt(b, a, y)

    tc, mdf_w, mnf_w = calculate_median_frequencies(emg, t, SAMPLE_FREQ, WINDOW_S, STEP_S)
    windows = pd.DataFrame({"t_center": tc, "MDF": mdf_w, "MNF": mnf_w})

    segments, peaks, valleys, smooth = segment_reps(emg, t, SAMPLE_FREQ)
    reps = per_rep_features(emg, t, SAMPLE_FREQ, segments)
    bounds = [0] + sorted(valleys) + [len(emg) - 1]
    dropped = pd.DataFrame([{"start_s": float(t[s]), "end_s": float(t[e]),
                             "duration_s": float(t[e] - t[s])}
                            for s, e in zip(bounds[:-1], bounds[1:])
                            if not (MIN_REP_S <= t[e] - t[s] <= MAX_REP_S)])

    # Bảng phân đoạn đầy đủ: mọi đoạn giữa hai biên, kể cả đoạn bị bỏ. Cột theo quy ước
    # data/processed/wsd_rep_boundaries.csv; thêm chỉ số mẫu trong FILE GỐC (fs gốc).
    kept = set(segments)
    to_raw = lambda i: int(i0 + round(i * fs / SAMPLE_FREQ))  # noqa: E731
    rows, k = [], 0
    for s, e in zip(bounds[:-1], bounds[1:]):
        ok = (s, e) in kept
        k += ok
        rows.append({"rep_idx": k if ok else pd.NA, "status": "rep" if ok else "dropped",
                     "rep_start_s": float(t[s]), "rep_end_s": float(t[e]),
                     "rep_duration_s": float(t[e] - t[s]),
                     "start_sample": to_raw(s), "end_sample": to_raw(e)})
    boundaries = pd.DataFrame(rows)
    boundaries["rep_idx"] = boundaries["rep_idx"].astype("Int64")
    boundaries["N_total"] = len(segments)
    boundaries["pipeline"] = PIPELINE_TAG
    assert k == len(segments) and len(boundaries) == len(valleys) + 1
    return {"emg": emg, "t": t, "smooth": smooth, "valleys": np.asarray(valleys),
            "windows": windows, "segments": segments, "reps": reps, "dropped": dropped,
            "boundaries": boundaries}


def summarize(res, n_video=None):
    w, reps = res["windows"], res["reps"]
    n_w, n_seg, n_rep = len(w), len(res["segments"]), len(reps)
    # assert cấu trúc TRƯỚC khi in bất kỳ con số nào
    assert n_w >= 3, f"chỉ có {n_w} cửa sổ 4 s; set quá ngắn để tính ρ"
    assert w[["MDF", "MNF"]].notna().all().all()
    assert n_rep <= n_seg and reps["rep_idx"].is_monotonic_increasing if n_rep else True
    assert len(res["dropped"]) + n_seg == len(res["valleys"]) + 1

    rho_w = float(spearmanr(w.t_center, w.MDF)[0])
    rho_w_mnf = float(spearmanr(w.t_center, w.MNF)[0])
    crit_w = critical_rho(n_w)
    ref = pd.read_csv(REF_W)
    assert len(ref) == 26
    n_more_neg = int((ref.MDF_spearman_rho < rho_w).sum())

    out = {"n_windows": n_w, "duration_s": float(w.t_center.iloc[-1] - w.t_center.iloc[0]),
           "rho_W_MDF": rho_w, "rho_W_MNF": rho_w_mnf, "crit_W": crit_w,
           "N_seg": n_seg, "n_rep_rows": n_rep, "n_dropped": len(res["dropped"])}
    lines = [
        "(W) ρ theo cửa sổ 4 s/2 s, toàn phổ — phép kiểm chính",
        f"  số cửa sổ           : {n_w}  (set dài {out['duration_s']:.1f} s giữa hai tâm cửa sổ)",
        f"  ρ_MDF               : {rho_w:+.3f}   (ρ_MNF {rho_w_mnf:+.3f})",
        f"  |ρ| tới hạn, n={n_w:<4}: {crit_w:.3f}",
        f"  ρ_MDF < 0           : {'có' if rho_w < 0 else 'KHÔNG'}",
        f"  |ρ_MDF| > tới hạn   : {'có' if abs(rho_w) > crit_w else 'KHÔNG'}",
        f"  so với Zenodo bicep : {n_more_neg}/26 trial có ρ_MDF âm hơn "
        f"(p10–p90 Zenodo: {np.percentile(ref.MDF_spearman_rho, 10):+.2f} … "
        f"{np.percentile(ref.MDF_spearman_rho, 90):+.2f})",
        "",
        "(R) cắt rep — chỉ đọc ρ theo rep khi N khớp video trong ngưỡng đã khai",
        f"  N bộ cắt            : {n_seg}   (đoạn bị bỏ vì ngoài [{MIN_REP_S}, {MAX_REP_S}] s: {len(res['dropped'])})",
    ]
    if n_video is not None:
        out["N_video"] = n_video
        lines.append(f"  N video             : {n_video}   (lệch {n_seg - n_video:+d})")
    else:
        lines.append("  N video             : chưa nhập (--n-video) → không đánh giá được bộ cắt")
    if n_rep >= 3:
        rho_r = float(spearmanr(reps.rep_idx, reps.MDF_mean)[0])
        out.update(rho_R_MDF=rho_r, crit_R=critical_rho(n_rep))
        lines += [f"  rep trung vị        : {reps.rep_duration_s.median():.2f} s "
                  f"(dài nhất {reps.rep_duration_s.max():.2f} s)",
                  f"  ρ_MDF theo rep      : {rho_r:+.3f}   (|ρ| tới hạn n={n_rep}: {out['crit_R']:.3f})"]
    return out, "\n".join(lines)


INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
S1, S2, REF = "#2a78d6", "#eb6834", "#9a9893"


def _style(ax):
    for a in np.atleast_1d(ax):
        a.set_facecolor(SURF)
        a.grid(True, color=GRID, lw=0.6)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color(INK2)
        a.tick_params(colors=INK2, labelsize=9)


def _draw_segmentation(ax, res, n_video=None):
    t, emg, sm = res["t"], res["emg"], res["smooth"]
    scale = np.abs(emg).max() / (sm.max() + 1e-12)
    ax.plot(t, emg, color=REF, lw=0.4, alpha=0.6)
    ax.plot(t[:len(sm)], sm * scale, color=S1, lw=2, label="đường bao RMS (đã chuẩn thang)")
    for v in res["valleys"]:
        ax.axvline(t[v], color=S2, lw=1, alpha=0.8)
    for _, d in res["dropped"].iterrows():
        ax.axvspan(d.start_s, d.end_s, color=S2, alpha=0.08, lw=0)
    nv = f", N video = {n_video}" if n_video is not None else ""
    ax.set_title(f"Cắt rep — N bộ cắt = {len(res['segments'])}{nv}; vạch cam = biên rep; "
                 f"nền cam = đoạn bị bỏ", color=INK, fontsize=11, loc="left")
    ax.set_xlabel("thời gian theo đồng hồ bản ghi (s)", color=INK2)
    ax.legend(loc="upper right", fontsize=9, frameon=False)


def plot_segmentation(res, path, title, n_video=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 4.5), facecolor=SURF)
    _style(ax)
    _draw_segmentation(ax, res, n_video)
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURF)
    plt.close(fig)


def plot(res, out, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(13, 10), facecolor=SURF,
                           gridspec_kw={"height_ratios": [1.3, 1, 0.55]})
    _style(ax)
    _draw_segmentation(ax[0], res, out.get("N_video"))

    w = res["windows"]
    ax[1].plot(w.t_center, w.MDF, color=S1, lw=2, marker="o", ms=4)
    ax[1].set_title(f"MDF theo cửa sổ 4 s — ρ = {out['rho_W_MDF']:+.3f}, n = {out['n_windows']}, "
                    f"|ρ| tới hạn = {out['crit_W']:.3f}", color=INK, fontsize=11, loc="left")
    ax[1].set_xlabel("thời gian (s)", color=INK2)
    ax[1].set_ylabel("MDF (Hz)", color=INK2)

    ref = pd.read_csv(REF_W).MDF_spearman_rho.values
    ax[2].scatter(ref, np.zeros_like(ref), s=60, color=REF, edgecolor=SURF, lw=1.5,
                  label="26 trial bicep Zenodo", zorder=2)
    ax[2].scatter([out["rho_W_MDF"]], [0], s=140, marker="D", color=S1, edgecolor=SURF,
                  lw=2, label="bản ghi này", zorder=3)
    ax[2].axvline(0, color=INK2, lw=0.8)
    ax[2].set_xlim(-1, 1)
    ax[2].set_yticks([])
    ax[2].set_xlabel("ρ_MDF theo cửa sổ", color=INK2)
    ax[2].set_title("Vị trí trong phân bố Zenodo bicep", color=INK, fontsize=11, loc="left")
    ax[2].legend(loc="upper right", fontsize=9, frameon=False, ncol=2)

    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURF)
    plt.close(fig)


def load_column(path, col, time_col, fs):
    with open(path, encoding="utf-8", errors="replace") as f:
        first = f.readline()
    if "|" in first or ":" in first.split(",")[0]:
        sys.exit(
            f"'{os.path.basename(path)}' trông như log Serial Monitor (dòng đầu: {first.strip()[:60]!r}),\n"
            "không phải luồng mẫu. Pipeline cần MỌI mẫu ở đúng --fs, dạng CSV có tiêu đề, ví dụ:\n"
            "    sample_idx,emg\n    0,-1234\n    1,-1198\n"
            "Xem mục 'Định dạng file' trong README.md.")
    df = pd.read_csv(path, sep=None, engine="python")
    if col is None:
        num = df.select_dtypes("number").columns
        num = [c for c in num if c != time_col]
        if len(num) != 1:
            sys.exit(f"File có {len(num)} cột số: {list(num)}. Chọn một cột bằng --col.")
        col = num[0]
    elif col not in df.columns and col.isdigit():
        col = df.columns[int(col)]
    if col not in df.columns:
        sys.exit(f"Không có cột '{col}'. Các cột: {list(df.columns)}")
    x = df[col].to_numpy(dtype=float)
    if time_col is not None:
        # đơn vị theo hậu tố tên cột: *_us, *_ms, còn lại là giây
        unit = 1e-6 if time_col.endswith("_us") else 1e-3 if time_col.endswith("_ms") else 1.0
        tt = df[time_col].to_numpy(dtype=np.float64) * unit
        fs_est = (len(tt) - 1) / (tt[-1] - tt[0])
        print(f"fs ước lượng từ cột '{time_col}': {fs_est:.2f} Hz (khai --fs {fs})")
        if abs(fs_est - fs) / fs > 0.01:
            sys.exit("fs thật lệch quá 1% so với --fs. Kiểm lại đơn vị cột thời gian hoặc cấu hình ADS1292R.")
    return x, col


def self_test():
    """Chạy run_pipeline trên trial Zenodo thô, assert khớp bảng tham chiếu; rồi thử đường
    resample bằng cách nâng chính trial đó lên 2000 Hz."""
    ref_w = pd.read_csv(REF_W)
    ref_r = pd.read_csv(REF_REPS, float_precision="round_trip")
    for s, tr, (tc, ec) in [(1, 5, (0, 1)), (3, 6, (2, 3))]:
        path = os.path.join(RAW_DIR, f"subject_{s}", f"trial_{tr}.csv")
        if not os.path.exists(path):
            sys.exit(f"Không có dữ liệu thô Zenodo {path}; self-test cần dataset/sEMG_data.")
        raw = pd.read_csv(path).values
        x, tz = raw[:, ec].astype(float), raw[:, tc].astype(float)

        res = run_pipeline(x, SAMPLE_FREQ, t_given=tz)
        out, _ = summarize(res)
        rw = ref_w[(ref_w.subject == s) & (ref_w.trial == tr)].iloc[0]
        rr = ref_r[(ref_r.subject == s) & (ref_r.trial == tr)].sort_values("rep_idx")
        assert out["n_windows"] == rw.n_windows
        assert abs(out["rho_W_MDF"] - rw.MDF_spearman_rho) < 1e-9, (out["rho_W_MDF"], rw.MDF_spearman_rho)
        assert out["N_seg"] == rr.N_total.iloc[0], (out["N_seg"], rr.N_total.iloc[0])
        got = res["reps"].set_index("rep_idx")
        assert list(got.index) == list(rr.rep_idx)
        assert np.allclose(got.MDF_mean.values, rr.MDF_mean.values, rtol=1e-9, atol=0)
        assert np.allclose(got.MNF_mean.values, rr.MNF_mean.values, rtol=1e-9, atol=0)
        print(f"S{s}/T{tr} @1259 Hz: khớp tham chiếu — n_windows={out['n_windows']}, "
              f"ρ_W={out['rho_W_MDF']:+.5f}, N={out['N_seg']}, {len(got)} rep MDF/MNF khớp v2b")

        # đường resample: 1259 → 2000 → (script) → 1259. Không kỳ vọng khớp bit, chỉ đo độ lệch.
        x2000 = signal.resample_poly(np.nan_to_num(x, nan=0.0), 2000, SAMPLE_FREQ)
        o2, _ = summarize(run_pipeline(x2000, 2000))
        print(f"  qua đường resample 2000→1259: ρ_W {o2['rho_W_MDF']:+.5f} "
              f"(lệch {o2['rho_W_MDF'] - out['rho_W_MDF']:+.5f}), N {o2['N_seg']} "
              f"(lệch {o2['N_seg'] - out['N_seg']:+d})")
    print("\nSelf-test PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?")
    ap.add_argument("--fs", type=float, help="tần số lấy mẫu của file (SPS), ví dụ 2000")
    ap.add_argument("--col", help="tên (hoặc số thứ tự) cột sEMG; bỏ trống nếu file chỉ có một cột số")
    ap.add_argument("--time-col", help="cột thời gian (giây), chỉ để kiểm fs thật")
    ap.add_argument("--start", type=float, default=None, help="giây bắt đầu rep đầu tiên (từ video)")
    ap.add_argument("--end", type=float, default=None, help="giây kết thúc rep cuối (từ video)")
    ap.add_argument("--n-video", type=int, default=None, help="số rep đếm từ video")
    ap.add_argument("--reps-only", action="store_true",
                    help="chỉ phân đoạn rep: ghi rep_boundaries.csv + segmentation.png, bỏ qua ρ")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not a.path or not a.fs:
        ap.error("cần đường dẫn file và --fs (hoặc --self-test)")

    x, col = load_column(a.path, a.col, a.time_col, a.fs)
    i0 = 0 if a.start is None else int(round(a.start * a.fs))
    i1 = len(x) if a.end is None else int(round(a.end * a.fs))
    if not (0 <= i0 < i1 <= len(x)):
        sys.exit(f"--start/--end ({a.start} … {a.end} s) nằm ngoài bản ghi: {len(x)} mẫu ở "
                 f"--fs {a.fs:g} chỉ dài {len(x) / a.fs:.1f} s. Kiểm lại --fs, hoặc firmware có ghi đủ mọi mẫu không.")
    if a.start is None or a.end is None:
        print("CHÚ Ý: chưa cắt theo video (--start/--end). Đoạn nghỉ đầu/cuối làm lệch ngưỡng cắt rep.")
    x = x[i0:i1]

    res = run_pipeline(x, a.fs, i0=i0)
    stem = os.path.splitext(os.path.basename(a.path))[0]
    od = os.path.join(os.path.dirname(os.path.abspath(a.path)), f"{stem}_out")
    os.makedirs(od, exist_ok=True)
    head = (f"file: {a.path}\ncột: {col}   fs gốc: {a.fs:g} SPS → {SAMPLE_FREQ} Hz\n"
            f"cắt: {a.start} … {a.end} s   (mọi thời gian tính theo đồng hồ bản ghi gốc)\n")
    res["boundaries"].to_csv(os.path.join(od, "rep_boundaries.csv"), index=False)

    if a.reps_only:
        bd = res["boundaries"]
        n_seg = len(res["segments"])
        lines = [f"N bộ cắt: {n_seg}   (đoạn bị bỏ vì ngoài [{MIN_REP_S}, {MAX_REP_S}] s: "
                 f"{int((bd.status == 'dropped').sum())})"]
        if a.n_video is not None:
            lines.append(f"N video : {a.n_video}   (lệch {n_seg - a.n_video:+d})")
        if n_seg:
            r = bd[bd.status == "rep"]
            lines.append(f"rep đầu {r.rep_start_s.iloc[0]:.2f} s, rep cuối kết thúc {r.rep_end_s.iloc[-1]:.2f} s; "
                         f"thời lượng rep trung vị {r.rep_duration_s.median():.2f} s")
        plot_segmentation(res, os.path.join(od, "segmentation.png"), stem, a.n_video)
        print(head)
        show = bd[["rep_idx", "status", "rep_start_s", "rep_end_s", "rep_duration_s",
                   "start_sample", "end_sample"]].copy()
        show[["rep_start_s", "rep_end_s", "rep_duration_s"]] = show[
            ["rep_start_s", "rep_end_s", "rep_duration_s"]].round(3)
        print(show.to_string(index=False, na_rep="-"))
        print()
        print("\n".join(lines))
        print(f"\n→ {od}")
        return 0

    out, text = summarize(res, a.n_video)
    res["windows"].to_csv(os.path.join(od, "windows.csv"), index=False)
    res["reps"].to_csv(os.path.join(od, "reps.csv"), index=False)
    res["dropped"].to_csv(os.path.join(od, "dropped.csv"), index=False)
    with open(os.path.join(od, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(head + "\n" + text + "\n")
    plot(res, out, os.path.join(od, "diagnostic.png"), stem)
    print(head)
    print(text)
    print(f"\n→ {od}")


if __name__ == "__main__":
    sys.exit(main())
