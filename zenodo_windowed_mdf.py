"""
MDF/MNF theo cửa sổ trượt — đúng biến thể đã sinh `trial_summary_with_flags.csv`
===============================================================================
Ba hàm dưới đây chép NGUYÊN VĂN logic từ `dataset/code.ipynb` (cell 1), vì notebook
không import được. Không sửa thân hàm: mọi con số tham chiếu trong
`reference/zenodo_bicep_window_rho.csv` đến từ đúng các dòng này.

Có HAI biến thể MDF theo cửa sổ 4 s / bước 2 s trong repo, và chúng KHÔNG cho cùng số:
  - toàn phổ dương, không cắt dải  → trial_summary_with_flags.csv   (file này)
  - cắt dải 20–450 Hz              → dataset/quality_metrics_v2.csv
Ví dụ S1/trial 5: ρ_MDF = −0.93558 (toàn phổ) so với −0.93595 (cắt dải).
`build_reference.py --check-raw` assert rằng file này tái lập trial_summary trên
3 trial bicep; nếu assert đó fail thì không được đặt số của mình cạnh bảng tham chiếu.

Tham số tính bằng GIÂY (4 s, 2 s) và nhân với sampling_rate bên trong, nên bản thân
hàm không phụ thuộc fs. Nhưng độ phân giải phổ = 1/4 s = 0.25 Hz chỉ giống Zenodo khi
cửa sổ dài đúng 4 s, và MDF/MNF lấy toàn phổ dương đến fs/2: ở 2000 SPS phổ kéo tới
1000 Hz thay vì 629.5 Hz, nên phải resample về 1259 Hz TRƯỚC khi gọi (xem README).
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import spearmanr

SAMPLE_FREQ = 1259  # Hz, Zenodo
WINDOW_S = 4
STEP_S = 2


def mean_frequency(power_spectrum, frequencies):
    if len(power_spectrum) == 0 or np.sum(power_spectrum) == 0:
        return None
    weighted_sum = np.sum(frequencies * power_spectrum)
    total_power = np.sum(power_spectrum)
    if total_power == 0:
        return None
    return weighted_sum / total_power


def median_frequency(power_spectrum, frequencies):
    cumulative_power = np.cumsum(power_spectrum)
    total_power = cumulative_power[-1]
    median_idx = np.where(cumulative_power >= (total_power / 2))[0][0]
    return frequencies[median_idx]


def calculate_median_frequencies(signal, time, sampling_rate, window_size_seconds, step_size_seconds):
    window_size = int(window_size_seconds * sampling_rate)
    step_size = int(step_size_seconds * sampling_rate)
    num_windows = int((len(signal) - window_size) // step_size + 1)

    median_frequencies = []
    mean_frequencies = []
    window_time_center = []

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window_signal = signal[start_idx:end_idx]
        window_time = time[start_idx:end_idx]

        center_time = window_time[int(len(window_time) // 2)]
        window_time_center.append(center_time)

        # Nguyên văn notebook: trừ trung bình của CẢ tín hiệu, không phải của cửa sổ.
        signal_mean = np.mean(signal)

        fft_result = fft(window_signal - signal_mean)
        frequencies = fftfreq(len(window_signal), d=1 / sampling_rate)
        power_spectrum = np.abs(fft_result) ** 2

        positive_frequencies = frequencies[: len(frequencies) // 2]
        positive_power_spectrum = power_spectrum[: len(power_spectrum) // 2]

        median_frequencies.append(median_frequency(positive_power_spectrum, positive_frequencies))
        mean_frequencies.append(mean_frequency(positive_power_spectrum, positive_frequencies))

    return window_time_center, median_frequencies, mean_frequencies


def window_rho(emg_filtered, time_arr, fs=SAMPLE_FREQ):
    """ρ Spearman của MDF và MNF theo thời gian, cùng định nghĩa cột *_spearman_rho
    trong trial_summary_with_flags.csv."""
    tc, mdf, mnf = calculate_median_frequencies(emg_filtered, time_arr, fs, WINDOW_S, STEP_S)
    return {
        "n_windows": len(tc),
        "duration_s": float(tc[-1] - tc[0]),
        "MDF_spearman_rho": float(spearmanr(tc, mdf)[0]),
        "MNF_spearman_rho": float(spearmanr(tc, mnf)[0]),
        "MDF_baseline": float(np.mean(mdf[:3])),
        "MDF_final": float(np.mean(mdf[-3:])),
    }
