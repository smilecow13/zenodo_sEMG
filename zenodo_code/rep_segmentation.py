"""
Rep Segmentation — nguồn DUY NHẤT của thuật toán phân đoạn rep
==============================================================
Tách từ cell 10 của `code.ipynb` để mọi script dùng chung một bản.

QUAN TRỌNG: notebook hiện vẫn giữ bản `segment_reps` riêng của nó. Hai bản
đang GIỐNG NHAU — đã kiểm chứng bằng cách đối chiếu số rep phát hiện được
với `per_cycle_features.csv` do notebook sinh ra: S1T5=86, S5T5=95,
S13T3=187 reps, khớp tuyệt đối cả tập `rep_idx`.

Nên thay cell 10 của notebook bằng:

    from rep_segmentation import segment_reps

Nếu để hai bản song song, chỉ cần một bên sửa tham số (prominence,
min_rep_s, smooth_ms) là `per_cycle_features_v2.csv` sẽ được sinh từ một
thuật toán khác với thuật toán notebook đang dùng — loại sai lệch rất khó
phát hiện vì không có lỗi nào được báo ra.

Đường bao hiện dùng là Moving RMS 200 ms + Savitzky-Golay, KHÔNG phải
Butterworth thông thấp 2–6 Hz; TKEO chưa được thử (xem ghi chú ở cuối file).
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter


def segment_reps(
    emg_filtered, time_arr, fs, min_rep_s=1.2, max_rep_s=4.5, smooth_ms=200
):
    """
    Phân đoạn rep bằng đường bao RMS + valley detection.

    Tham số min=1.2s / max=4.5s theo baseline 30 BPM cộng velocity loss.
    Valley được tìm trực tiếp trên `-smooth` để tránh các đỉnh giả sát nhau.

    Parameters
    ----------
    emg_filtered : array-like
        Tín hiệu sEMG đã lọc bandpass 20–450 Hz.
    time_arr : array-like
        Trục thời gian (giây), cùng độ dài với emg_filtered.
    fs : float
        Tần số lấy mẫu (Hz).
    min_rep_s, max_rep_s : float
        Khoảng thời lượng rep hợp lệ. Đoạn ngoài khoảng này bị loại.
    smooth_ms : float
        Độ rộng cửa sổ Moving RMS (ms).

    Returns
    -------
    segments : list[tuple[int, int]]
        Các cặp (start_idx, end_idx) của rep hợp lệ.
    peaks : np.ndarray
    valleys : np.ndarray
    smooth : np.ndarray
        Đường bao đã làm mượt (ngắn hơn tín hiệu gốc vài mẫu).
    """
    win = int(smooth_ms * 1e-3 * fs)

    # Moving RMS dạng vectorized qua cumsum
    sig_sq = np.pad(emg_filtered**2, win // 2, mode="edge")
    cumsum = np.cumsum(sig_sq)
    rms_env = np.sqrt((cumsum[win:] - cumsum[:-win]) / win)

    sg_win = max(win * 3, 11)
    if sg_win % 2 == 0:
        sg_win += 1
    smooth = savgol_filter(rms_env, window_length=sg_win, polyorder=3)

    min_dist = int(min_rep_s * fs)
    prom = np.std(smooth) * 0.5

    peaks, _ = find_peaks(smooth, distance=min_dist, prominence=prom)
    valleys, _ = find_peaks(-smooth, distance=min_dist, prominence=prom)

    boundary = sorted(valleys)
    if not boundary:
        return [], peaks, valleys, smooth

    all_bounds = [0] + list(boundary) + [len(emg_filtered) - 1]
    segments = [
        (s, e)
        for s, e in zip(all_bounds[:-1], all_bounds[1:])
        if min_rep_s <= (time_arr[e] - time_arr[s]) <= max_rep_s
    ]

    return segments, peaks, valleys, smooth


# ── Ghi chú về lựa chọn đường bao (cho phần Methods của bài báo) ────────────
# Bốn phương án thông dụng: (1) Butterworth thông thấp 2–6 Hz, (2) Moving RMS
# 50–200 ms, (3) Hilbert, (4) TKEO rồi chỉnh lưu. Pipeline này dùng (2).
# TKEO:  ψ[n] = x[n]² − x[n−1]·x[n+1]
# xấp xỉ tích biên độ² với tần số², nên khuếch đại đúng lúc cơ hoạt hoá và
# dìm nhiễu nền biên độ thấp — chỉ 3 mẫu, 2 phép nhân, gần như miễn phí trên
# vi điều khiển. Đáng thử như một nhánh đối chứng cho bước phân đoạn onset,
# nhưng hiện CHƯA được triển khai ở đâu trong pipeline.
