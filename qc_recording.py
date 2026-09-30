"""
Kiểm chất lượng một bản ghi sEMG TRƯỚC khi phân tích
====================================================
Chạy trước analyze_recording.py / model_direction.py. Script chỉ in ra terminal, không ghi file.

    python qc_recording.py recordings/<file>.csv --fs 2000 \\
        --col ch1_raw24 --col2 ch2_raw24

Mặc định khớp định dạng firmware hiện tại:
  sample_idx,timestamp_us,ch1_raw24,ch2_raw24,raw_emg16,filtered_emg16,lead_off
Cột nào vắng thì mục kiểm tương ứng được bỏ qua và báo rõ.

Các mục kiểm:
  [1] bộ đếm mẫu liên tục (không rơi, không lặp)              — cột --idx-col
  [2] fs thật từ timestamp, lệch ≤ 1% so với --fs              — cột --time-col
  [3] độ dài bản ghi, số cửa sổ 4 s, |ρ| tới hạn tương ứng
  [4] bão hoà ADC (chạm ±2²³) và % thang đo đã dùng
  [5] lead-off                                                 — cột --leadoff-col
  [6] tỉ lệ công suất ở vạch điện lưới 50·k Hz, so với Zenodo bicep
  [7] MDF cả bản ghi theo pipeline Zenodo, và MDF sau khi bỏ vạch điện lưới (chỉ chẩn đoán)
  [8] (nếu có --col2) hai kênh có mang cùng một tín hiệu không

Tín hiệu cho [6]–[8] đi qua đúng bước đầu của pipeline: resample về 1259 Hz, Butterworth
bậc 4 20–450 Hz, filtfilt. Mức Zenodo ở [6] được tính bằng CHÍNH hàm mains_share dưới đây
(build_reference.py --mains), cùng fs, cùng Welch, nên hai con số đặt cạnh nhau được.

Ngưỡng ĐẠT/CHƯA ĐẠT: các mục cấu trúc [1] [2] [4] [5] là điều kiện cứng. Ngưỡng của [3] và [6]
là ĐỀ XUẤT (MIN_DURATION_S, MAINS_MAX), phải chốt trong mục "Khai trước" của README trước
buổi đo chính thức. [7] [8] chỉ để chẩn đoán, không có ngưỡng.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "zenodo_code")]

from analyze_recording import critical_rho, to_zenodo_rate  # noqa: E402
from zenodo_windowed_mdf import SAMPLE_FREQ, STEP_S, WINDOW_S  # noqa: E402

REF_MAINS = os.path.join(HERE, "reference", "zenodo_bicep_mains_share.csv")
FULL_SCALE = 2 ** 23          # ADS1292R 24-bit
FS_TOL = 0.01
MIN_DURATION_S = 60.0         # ĐỀ XUẤT: dưới mức này < 30 cửa sổ, |ρ| tới hạn > 0.36
MAINS_MAX = 0.10              # ĐỀ XUẤT: Zenodo bicep 3.5–4.4%; phổ phẳng không nhiễu ≈ 4.2%
MAINS_HW = 1.0                # nửa độ rộng vạch điện lưới (Hz)
NPERSEG = 8192                # ở 1259 Hz: độ phân giải 0.15 Hz


def pipeline_filter(x, fs):
    y = to_zenodo_rate(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0), fs)
    b, a = signal.butter(4, [20, 450], btype="bandpass", fs=SAMPLE_FREQ)
    return signal.filtfilt(b, a, y)


def _mains_mask(f):
    m = np.zeros_like(f, dtype=bool)
    for h in range(50, 451, 50):
        m |= np.abs(f - h) <= MAINS_HW
    return m


def mains_share(y):
    """Tỉ lệ công suất Welch trong dải 20–450 Hz nằm ở các vạch 50, 100, …, 450 Hz (±1 Hz).
    y: tín hiệu đã qua pipeline_filter, ở 1259 Hz."""
    f, p = signal.welch(y, fs=SAMPLE_FREQ, nperseg=min(NPERSEG, len(y)))
    band = (f >= 20) & (f <= 450)
    return float(p[band & _mains_mask(f)].sum() / p[band].sum())


def mdf_welch(y, drop_mains=False):
    f, p = signal.welch(y, fs=SAMPLE_FREQ, nperseg=min(NPERSEG, len(y)))
    if drop_mains:
        p = np.where(_mains_mask(f), 0.0, p)
    band = (f >= 20) & (f <= 450)
    c = np.cumsum(p[band])
    return float(f[band][np.searchsorted(c, c[-1] / 2)])


def notch_mains(y):
    for h in range(50, 451, 50):
        bn, an = signal.iirnotch(h, 30, SAMPLE_FREQ)
        y = signal.filtfilt(bn, an, y)
    return y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--fs", type=float, required=True, help="tần số lấy mẫu khai báo (SPS)")
    ap.add_argument("--col", default="ch1_raw24", help="kênh sEMG chính")
    ap.add_argument("--col2", default=None, help="kênh thứ hai, để kiểm [8]")
    ap.add_argument("--idx-col", default="sample_idx")
    ap.add_argument("--time-col", default="timestamp_us")
    ap.add_argument("--time-unit", default="us", choices=["s", "ms", "us"])
    ap.add_argument("--leadoff-col", default="lead_off")
    a = ap.parse_args()

    d = pd.read_csv(a.path)
    if a.col not in d.columns:
        sys.exit(f"Không có cột '{a.col}'. Các cột: {list(d.columns)}")
    fails, rows = [], []

    def report(tag, ok, text):
        mark = {True: "ĐẠT     ", False: "CHƯA ĐẠT", None: "—       "}[ok]
        rows.append(f"{tag:5s} {mark} {text}")
        if ok is False:
            fails.append(tag)

    # [1] bộ đếm mẫu
    if a.idx_col in d.columns:
        di = np.diff(d[a.idx_col].to_numpy())
        miss = int((di[di > 1] - 1).sum())
        back = int((di <= 0).sum())
        report("[1]", miss == 0 and back == 0,
               f"bộ đếm mẫu: {len(d)} mẫu, thiếu {miss}, lặp/lùi {back}")
    else:
        report("[1]", None, f"bỏ qua: không có cột '{a.idx_col}'")

    # [2] fs thật
    if a.time_col in d.columns:
        scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6}[a.time_unit]
        ts = d[a.time_col].to_numpy(dtype=np.float64) * scale
        span = ts[-1] - ts[0]
        n_int = (d[a.idx_col].iloc[-1] - d[a.idx_col].iloc[0]) if a.idx_col in d.columns else len(d) - 1
        fs_real = n_int / span
        report("[2]", abs(fs_real - a.fs) / a.fs <= FS_TOL,
               f"fs thật {fs_real:.2f} Hz so với khai báo {a.fs:g} (lệch {100 * (fs_real / a.fs - 1):+.2f}%, "
               f"ngưỡng ±{100 * FS_TOL:.0f}%)")
    else:
        report("[2]", None, f"bỏ qua: không có cột '{a.time_col}'")

    # [3] độ dài
    dur = len(d) / a.fs
    n_win = int((dur - WINDOW_S) // STEP_S + 1) if dur >= WINDOW_S else 0
    crit = f"{critical_rho(n_win):.2f}" if n_win > 2 else "—"
    report("[3]", dur >= MIN_DURATION_S,
           f"độ dài {dur:.1f} s → {n_win} cửa sổ 4 s, |ρ| tới hạn {crit} "
           f"(ngưỡng đề xuất ≥ {MIN_DURATION_S:.0f} s; Zenodo bicep trung vị 262 s)")

    # [4] bão hoà
    x = d[a.col].to_numpy(dtype=float)
    peak = np.nanmax(np.abs(x))
    n_sat = int((np.abs(x) >= FULL_SCALE - 1).sum())
    report("[4]", n_sat == 0, f"bão hoà: {n_sat} mẫu chạm ±2²³; đỉnh dùng {100 * peak / FULL_SCALE:.2f}% thang đo")

    # [5] lead-off
    if a.leadoff_col in d.columns:
        n_lo = int((d[a.leadoff_col] != 0).sum())
        report("[5]", n_lo == 0, f"lead-off: {n_lo} mẫu báo tuột điện cực")
    else:
        report("[5]", None, f"bỏ qua: không có cột '{a.leadoff_col}'")

    # [6] điện lưới
    y = pipeline_filter(x, a.fs)
    share = mains_share(y)
    if os.path.exists(REF_MAINS):
        ref = pd.read_csv(REF_MAINS)["mains_share"]
        zr = f"Zenodo bicep {100 * ref.min():.1f}–{100 * ref.max():.1f}%"
    else:
        zr = "chưa có mốc Zenodo: chạy python build_reference.py --mains"
    report("[6]", share <= MAINS_MAX,
           f"nhiễu điện lưới: {100 * share:.1f}% công suất 20–450 Hz nằm ở vạch 50·k Hz "
           f"({zr}; ngưỡng đề xuất ≤ {100 * MAINS_MAX:.0f}%)")

    # [7] MDF — chẩn đoán
    report("[7]", None, f"MDF cả bản ghi theo pipeline {mdf_welch(y):.1f} Hz; bỏ vạch điện lưới "
                        f"{mdf_welch(y, drop_mains=True):.1f} Hz (bicep Zenodo đầu set ~80–110 Hz; chỉ chẩn đoán)")

    # [8] hai kênh — chẩn đoán
    if a.col2:
        if a.col2 not in d.columns:
            sys.exit(f"Không có cột '{a.col2}'.")
        y2 = pipeline_filter(d[a.col2].to_numpy(dtype=float), a.fs)
        f, coh = signal.coherence(y, y2, fs=SAMPLE_FREQ, nperseg=2048)
        clean = (f > 60) & (f < 95)
        r = np.corrcoef(notch_mains(y), notch_mains(y2))[0, 1]
        report("[8]", None, f"{a.col} ↔ {a.col2}: coherence trung vị 60–95 Hz {np.median(coh[clean]):.3f}, "
                            f"tương quan sau khi bỏ điện lưới {r:.3f} (gần 1 = hai kênh mang CÙNG tín hiệu)")

    print(f"file: {a.path}\nkênh: {a.col}   fs khai báo: {a.fs:g} SPS\n")
    print("\n".join(rows))
    print()
    if fails:
        print(f"CHƯA ĐỦ ĐIỀU KIỆN phân tích: {', '.join(fails)}. "
              "Ngưỡng [3] [6] là đề xuất — chốt trong README trước buổi đo.")
        return 1
    print("Đủ điều kiện để chạy analyze_recording.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
