"""
Extra Feature Extraction Module (mục 5.3 của kế hoạch)
=======================================================
Bổ sung các đặc trưng còn thiếu so với mục 5.3, khớp 1:1 với sơ đồ
per-rep/3-thirds mà notebook `code.ipynb` (cell 11) đang dùng.

Đặc trưng MỚI (14 đại lượng gốc × 3 thống kê max/min/mean = 42 cột):

  Miền thời gian (7 — RMS đã có sẵn trong notebook):
    MAV    Mean absolute value            mean(|x|)
    iEMG   Integrated EMG                 Σ|x|·dt                      [V·s]
    WL     Waveform length                Σ|x[n+1]−x[n]|               [V]
    ZC     Zero crossings                 có ngưỡng chống nhiễu  [lần/giây]
    SSC    Slope sign changes             có ngưỡng chống nhiễu  [lần/giây]
    WAMP   Willison amplitude             có ngưỡng              [lần/giây]
    DASDV  Diff. abs. std. dev. value     sqrt(mean(Δx²))

  Miền tần số (7 — MNF/MDF/TP đã có sẵn):
    PKF      Peak frequency              argmax P(f)
    LFHF     Tỷ lệ công suất             P[20–50 Hz] / P[50–450 Hz]
    FInsm5   Chỉ số phổ Dimitrov         ∫f⁻¹P df / ∫f⁵P df
    SpecEnt  Spectral entropy            chuẩn hoá về [0,1]
    SM1..SM3 Spectral moments bậc 1–3    ∫fⁿP(f) df

Lưu ý SM0 bị bỏ vì trùng với cột TP đã có trong `per_cycle_features.csv`.


QUY ƯỚC & CÁC ĐÁNH ĐỔI ĐÃ CHỐT (đọc trước khi dùng)
----------------------------------------------------
1. ƯỚC LƯỢNG PSD. Mục 5.3 yêu cầu Welch (Hamming 256 ms, overlap 50%),
   nhưng notebook hiện dùng periodogram FFT thuần không cửa sổ. Module
   cho cả hai qua `psd_method`, MẶC ĐỊNH là 'fft' để MNF/MDF tính lại
   trùng khít với `per_cycle_features.csv` hiện có — tránh làm lệch kết
   quả GO 16.8% đã chốt. Đổi sang 'welch' là một nhánh ablation riêng,
   không phải mặc định im lặng.

2. CHUẨN HOÁ THEO ĐỘ DÀI. Notebook chia mỗi rep thành 3 phần có độ dài
   KHÁC NHAU giữa các rep (vì thời lượng rep thay đổi khi mỏi). Do đó:
   - ZC, SSC, WAMP trả về dạng TẦN SỐ SỰ KIỆN (lần/giây), không phải số
     đếm thô — nếu để số đếm thô thì feature sẽ tỷ lệ với độ dài đoạn và
     lẫn lộn với thông tin thời lượng rep.
   - SpecEnt chuẩn hoá bằng ln(số bin) để không phụ thuộc độ dài đoạn.
   - PSD mặc định ở dạng mật độ (V²/Hz, `density=True`) nên SM1–SM3 so
     sánh được giữa các đoạn dài ngắn khác nhau.
   - iEMG và WL theo ĐỊNH NGHĨA là đại lượng tích luỹ nên vẫn phụ thuộc
     độ dài đoạn. Giữ đúng định nghĩa y văn, nhưng phải biết rằng chúng
     đồng biến với `rep_duration_s` — khi đọc SHAP/importance cần tách
     bạch, đừng kết luận "iEMG dự báo mỏi" nếu thực chất là thời lượng.

3. BẤT BIẾN THEO THANG PSD. PKF, LFHF, FInsm5, SpecEnt, MNF, MDF là các
   đại lượng TỶ SỐ nên không đổi khi PSD bị nhân hằng số — tức chúng
   giống nhau dù chọn `density` True hay False. Chỉ SM1–SM3 bị ảnh hưởng.

4. NGƯỠNG CHỐNG NHIỄU. ZC/SSC/WAMP cần ngưỡng. Ngưỡng tuyệt đối cố định
   là sai ở đây vì gain giữa các subject khác nhau (chính lý do N1 mang
   lại +12.78%). Nên ngưỡng được tính MỘT LẦN CHO MỖI TRIAL theo σ của
   tín hiệu đã lọc (`compute_trial_thresholds`). Không dùng σ của từng
   đoạn nhỏ, vì như vậy ngưỡng sẽ tự trôi theo biên độ khi mỏi và triệt
   tiêu đúng cái tín hiệu ta muốn đo.

5. FInsm5 CÓ ĐỘ LỚN RẤT NHỎ (cỡ 1e-14…1e-16) do mẫu số chứa f⁵ với f tới
   450 Hz. float64 xử lý thoải mái, và z-score N1 phía sau sẽ đưa về
   thang dùng được. Chỉ cần đừng bất ngờ khi in ra thấy số bé.

6. PHA ECC/CON. Mục 5.3 muốn tính riêng cho pha concentric / eccentric.
   Module này theo đúng sơ đồ hiện có của notebook (chia 3 phần đều rồi
   lấy max/min/mean), CHƯA tách ecc/con — việc đó cần IMU hoặc thuật toán
   phát hiện pha riêng, là một bước độc lập.

Usage
-----
    from extra_features import (
        compute_trial_thresholds, extract_extra_per_rep,
        merge_extra_features, EXTRA_BASE_COLS,
    )

    # trong notebook, sau khi đã có `filtered`, `raw_time`, `segments`:
    thr = compute_trial_thresholds(filtered)
    extra_df = extract_extra_per_rep(
        filtered, raw_time, sample_freq, segments,
        subject=subject_number, trial=trial, thresholds=thr)

    # gộp vào per_cycle_features.csv (khớp theo subject/trial/rep_idx)
    df_full = merge_extra_features(per_cycle_df, extra_all_df)
"""

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import welch

# Công thức N1 chỉ tồn tại MỘT bản duy nhất, ở n1_features. Bản ở đây từng là
# bản sao và đã dùng epsilon tuyệt đối 1e-9 — đủ để vô hiệu hoá chuẩn hoá
# phương sai của FInsm5 (σ baseline ~2.2e-13). Import lại để không tái diễn.
from n1_features import apply_N1_zscore

# ── Cấu hình băng tần (theo mục 5.3) ────────────────────────────────────────
EMG_BAND = (20.0, 450.0)  # trùng với bandpass Butterworth bậc 4 của pipeline
LF_BAND = (20.0, 50.0)  # dải thấp — mục 5.3 ghi rõ 20–50 Hz
HF_BAND = (50.0, 450.0)  # dải cao  — mục 5.3 ghi rõ 50–450 Hz

# ── Hệ số ngưỡng chống nhiễu (nhân với σ của tín hiệu đã lọc, theo trial) ───
# Đây là các giá trị khởi điểm hợp lý, KHÔNG phải hằng số thiêng liêng:
# nên kiểm tra lại bằng cách vẽ histogram ZC/WAMP và soi vài rep cụ thể.
K_ZC = 0.01  # ngưỡng nhỏ, chỉ để loại dao động nhiễu quanh mức 0
K_SSC = 0.01
K_WAMP = 0.10  # Willison truyền thống ~50 µV trên EMG thô; 0.1σ là tương tự

# ── Welch (chỉ dùng khi psd_method='welch', theo đặc tả mục 5.3) ────────────
WELCH_WINDOW_S = 0.256  # Hamming 256 ms
WELCH_OVERLAP = 0.5  # overlap 50%

# ── Tên 14 đại lượng gốc mới + 42 cột thống kê ──────────────────────────────
EXTRA_TIME_NAMES = ["MAV", "iEMG", "WL", "ZC", "SSC", "WAMP", "DASDV"]
EXTRA_FREQ_NAMES = ["PKF", "LFHF", "FInsm5", "SpecEnt", "SM1", "SM2", "SM3"]
EXTRA_BASE_NAMES = EXTRA_TIME_NAMES + EXTRA_FREQ_NAMES

EXTRA_BASE_COLS = [
    f"{name}_{stat}" for name in EXTRA_BASE_NAMES for stat in ("max", "min", "mean")
]

MERGE_KEYS = ["subject", "trial", "rep_idx"]

_EPS = 1e-12


# ═══════════════════════════════════════════════════════════════════════════
# PSD
# ═══════════════════════════════════════════════════════════════════════════
def compute_psd(seg, fs, band=EMG_BAND, psd_method="fft", density=True):
    """
    Ước lượng phổ công suất một phía, đã giới hạn trong `band`.

    psd_method='fft' tái hiện ĐÚNG cách notebook cell 11 đang làm:
    trừ trung bình đoạn, FFT, rồi lấy mask (freqs >= lo) & (freqs <= hi).
    Mask này tự loại phần tần số âm của `fftfreq` nên không cần cắt nửa phổ.

    psd_method='welch' theo đặc tả mục 5.3 (Hamming 256 ms, overlap 50%),
    giảm phương sai ước lượng nhưng cho giá trị MNF/MDF khác 'fft' → dùng
    như một nhánh ablation, đừng trộn lẫn hai phương pháp trong cùng bảng.

    Parameters
    ----------
    seg : array-like
        Đoạn tín hiệu sEMG ĐÃ LỌC bandpass.
    fs : float
        Tần số lấy mẫu (Hz).
    band : tuple(float, float)
        Dải tần giữ lại.
    psd_method : {'fft', 'welch'}
    density : bool
        True → PSD dạng mật độ (V²/Hz), so sánh được giữa các đoạn dài
        ngắn khác nhau. Chỉ ảnh hưởng SM1–SM3; các feature tỷ số bất biến.

    Returns
    -------
    freqs : np.ndarray
    pxx : np.ndarray
        Rỗng nếu đoạn quá ngắn hoặc không có bin nào trong băng.
    """
    seg = np.asarray(seg, dtype=float)
    n = len(seg)
    if n < 16:
        return np.array([]), np.array([])

    if psd_method == "welch":
        nperseg = min(int(WELCH_WINDOW_S * fs), n)
        freqs, pxx = welch(
            seg - seg.mean(),
            fs=fs,
            window="hamming",
            nperseg=nperseg,
            noverlap=int(nperseg * WELCH_OVERLAP),
            scaling="density" if density else "spectrum",
        )
    elif psd_method == "fft":
        spec = fft(seg - seg.mean())
        freqs = fftfreq(n, 1.0 / fs)
        pxx = np.abs(spec) ** 2
        if density:
            pxx = pxx / (fs * n)  # → V²/Hz
    else:
        raise ValueError(f"psd_method phải là 'fft' hoặc 'welch', nhận được {psd_method!r}")

    mask = (freqs >= band[0]) & (freqs <= band[1])
    return freqs[mask], pxx[mask]


def _delta_f(freqs):
    """Bước tần số. Trả 0.0 nếu không đủ bin để xác định."""
    return float(freqs[1] - freqs[0]) if len(freqs) >= 2 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Miền thời gian
# ═══════════════════════════════════════════════════════════════════════════
def mav(x):
    """Mean absolute value: mean(|x|)."""
    return float(np.mean(np.abs(x)))


def iemg(x, fs):
    """
    Integrated EMG: Σ|x|·dt  [V·s].

    CẢNH BÁO: là đại lượng tích luỹ nên tỷ lệ với độ dài đoạn. Xem mục 2
    của docstring module.
    """
    return float(np.sum(np.abs(x)) / fs)


def waveform_length(x):
    """
    Waveform length: Σ|x[n+1] − x[n]|  [V].

    Cùng cảnh báo tích luỹ như iEMG.
    """
    return float(np.sum(np.abs(np.diff(x))))


def zero_crossings(x, fs, threshold):
    """
    Zero crossings CÓ NGƯỠNG, trả về số lần/giây.

    Chỉ tính khi vừa đổi dấu VÀ biên độ bước nhảy ≥ threshold — điều kiện
    thứ hai là phần "chống nhiễu" mà mục 5.3 yêu cầu, loại bỏ các lần
    cắt 0 giả do nhiễu biên độ thấp.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    sign_change = np.sign(x[:-1]) != np.sign(x[1:])
    big_enough = np.abs(x[:-1] - x[1:]) >= threshold
    count = int(np.sum(sign_change & big_enough))
    return count * fs / len(x)


def slope_sign_changes(x, fs, threshold):
    """
    Slope sign changes CÓ NGƯỠNG, trả về số lần/giây.

    Đếm các điểm cực trị cục bộ mà độ "nhô" vượt threshold.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return 0.0
    prev_d = x[1:-1] - x[:-2]
    next_d = x[1:-1] - x[2:]
    count = int(np.sum((prev_d * next_d) >= threshold))
    return count * fs / len(x)


def willison_amplitude(x, fs, threshold):
    """Willison amplitude: số lần |Δx| > threshold, trả về số lần/giây."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    count = int(np.sum(np.abs(np.diff(x)) > threshold))
    return count * fs / len(x)


def dasdv(x):
    """
    Difference absolute standard deviation value:
        sqrt( Σ(x[n+1] − x[n])² / (N−1) )

    Bất biến với độ dài đoạn (đã chia N−1).
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    d = np.diff(x)
    return float(np.sqrt(np.sum(d**2) / len(d)))


# ═══════════════════════════════════════════════════════════════════════════
# Miền tần số
# ═══════════════════════════════════════════════════════════════════════════
def mean_frequency(freqs, pxx):
    """
    MNF = Σ f·P(f) / Σ P(f)

    Có ở đây để tính được MNF dưới MỘT phương pháp ước lượng phổ BẤT KỲ
    (fft hay welch), phục vụ nhánh đối chứng Welch. Các cột MNF_* trong
    `per_cycle_features.csv` do notebook sinh ra bằng đường fft — hàm này
    cho kết quả trùng khít khi gọi với psd_method='fft'.
    """
    if len(pxx) == 0:
        return np.nan
    total = float(np.sum(pxx))
    if total <= _EPS:
        return np.nan
    return float(np.sum(freqs * pxx) / total)


def median_frequency(freqs, pxx):
    """MDF: tần số mà công suất tích luỹ đạt 50% tổng công suất."""
    if len(pxx) == 0:
        return np.nan
    cum = np.cumsum(pxx)
    if cum[-1] <= _EPS:
        return np.nan
    idx = int(np.searchsorted(cum, cum[-1] / 2.0))
    return float(freqs[min(idx, len(freqs) - 1)])


def spectral_features(freqs, pxx):
    """
    Toàn bộ 10 đặc trưng miền tần số của mục 5.3 trên một PSD đã cho.

    Gồm cả bộ kinh điển MNF/MDF/TP để nhánh Welch có thể tính lại chúng —
    các cột MNF/MDF/TP trong v1/v2 đến từ notebook (đường fft), còn hàm này
    cho phép tính cùng đại lượng dưới phương pháp ước lượng phổ khác.
    """
    return {
        "MNF": mean_frequency(freqs, pxx),
        "MDF": median_frequency(freqs, pxx),
        "TP": float(np.sum(pxx)) if len(pxx) else np.nan,
        "PKF": peak_frequency(freqs, pxx),
        "LFHF": lf_hf_ratio(freqs, pxx),
        "FInsm5": finsm5(freqs, pxx),
        "SpecEnt": spectral_entropy(freqs, pxx),
        "SM1": spectral_moment(freqs, pxx, 1),
        "SM2": spectral_moment(freqs, pxx, 2),
        "SM3": spectral_moment(freqs, pxx, 3),
    }


def peak_frequency(freqs, pxx):
    """PKF: tần số tại đỉnh phổ công suất."""
    if len(pxx) == 0:
        return np.nan
    return float(freqs[int(np.argmax(pxx))])


def lf_hf_ratio(freqs, pxx, lf_band=LF_BAND, hf_band=HF_BAND):
    """
    Tỷ lệ công suất dải thấp / dải cao (mục 5.3: 20–50 Hz / 50–450 Hz).

    Khi mỏi, phổ dịch về tần số thấp → LF/HF tăng. Bất biến với thang PSD.
    Biên 50 Hz được tính vào dải cao (nửa khoảng [20,50) và [50,450]) để
    không đếm trùng một bin vào cả hai dải.
    """
    if len(pxx) == 0:
        return np.nan
    lf = float(np.sum(pxx[(freqs >= lf_band[0]) & (freqs < lf_band[1])]))
    hf = float(np.sum(pxx[(freqs >= hf_band[0]) & (freqs <= hf_band[1])]))
    return lf / (hf + _EPS)


def spectral_moment(freqs, pxx, order):
    """
    Spectral moment bậc n:  SMn = ∫ fⁿ P(f) df ≈ Σ fᵢⁿ Pᵢ Δf

    SM0 (= tổng công suất) bị bỏ khỏi module vì trùng cột TP đã có.
    KHÔNG bất biến với thang PSD — xem mục 3 của docstring module.
    """
    if len(pxx) == 0:
        return np.nan
    return float(np.sum((freqs**order) * pxx) * _delta_f(freqs))


def finsm5(freqs, pxx):
    """
    Chỉ số phổ Dimitrov bậc 5:

        FInsm5 = ∫ f⁻¹ P(f) df  /  ∫ f⁵ P(f) df

    Mục 5.3 đánh giá "nhạy với mỏi hơn MDF nhiều, bắt buộc đưa vào":
    tử số khuếch đại phần năng lượng tần thấp (tăng khi mỏi), mẫu số
    khuếch đại phần tần cao (giảm khi mỏi) → tỷ số tăng mạnh theo mỏi,
    nhạy hơn MDF vốn chỉ là một điểm chia đôi phổ.

    Δf triệt tiêu giữa tử và mẫu nên giá trị bất biến với thang PSD.
    Độ lớn điển hình rất nhỏ (~1e-14…1e-16) — xem mục 5 của docstring.
    """
    if len(pxx) == 0:
        return np.nan
    safe_f = np.where(freqs > _EPS, freqs, np.nan)
    num = np.nansum(pxx / safe_f)
    den = np.nansum((freqs**5) * pxx)
    if den <= _EPS:
        return np.nan
    return float(num / den)


def spectral_entropy(freqs, pxx):
    """
    Spectral entropy đã chuẩn hoá về [0,1]:

        p = P / ΣP,   H = −Σ p·ln(p) / ln(số bin)

    Chia cho ln(số bin) để không phụ thuộc độ dài đoạn (mục 2 docstring).
    Gần 1 = phổ phẳng/giống nhiễu; nhỏ = năng lượng tập trung hẹp.
    """
    if len(pxx) < 2:
        return np.nan
    total = float(np.sum(pxx))
    if total <= _EPS:
        return np.nan
    p = pxx / total
    p = p[p > _EPS]
    if len(p) == 0:
        return np.nan
    return float(-np.sum(p * np.log(p)) / np.log(len(pxx)))


# ═══════════════════════════════════════════════════════════════════════════
# Ngưỡng theo trial
# ═══════════════════════════════════════════════════════════════════════════
def compute_trial_thresholds(emg_filtered, k_zc=K_ZC, k_ssc=K_SSC, k_wamp=K_WAMP):
    """
    Tính ngưỡng chống nhiễu MỘT LẦN cho cả trial, theo σ tín hiệu đã lọc.

    Cố ý KHÔNG tính theo từng đoạn nhỏ: nếu ngưỡng trôi theo biên độ của
    từng đoạn thì khi mỏi (biên độ tăng) ngưỡng cũng tăng theo và triệt
    tiêu đúng hiệu ứng ta cần đo. Xem mục 4 của docstring module.

    Lưu ý SSC so sánh với tích hai độ dốc (đơn vị V²) nên ngưỡng của nó
    được bình phương tương ứng để cùng đơn vị.

    Returns
    -------
    dict : {'zc': float, 'ssc': float, 'wamp': float}
    """
    sigma = float(np.std(np.asarray(emg_filtered, dtype=float)))
    return {
        "zc": k_zc * sigma,
        "ssc": (k_ssc * sigma) ** 2,
        "wamp": k_wamp * sigma,
    }


def compute_baseline_thresholds(emg_filtered, rep_segments, baseline_n=3,
                                k_zc=K_ZC, k_ssc=K_SSC, k_wamp=K_WAMP):
    """
    Ngưỡng chống nhiễu từ CHỈ `baseline_n` rep segment đầu của trial.

    `compute_trial_thresholds` lấy σ của toàn trial, tức gồm cả những rep
    chưa xảy ra tại thời điểm suy luận. Biên độ sEMG tăng khi mỏi nên σ_trial
    mang thông tin về N_total, mà FCF = rep_idx/N_total. Hàm này ghép đúng
    `baseline_n` đoạn đầu rồi lấy std (cùng ddof=0 như hàm cũ), nên tính
    được trên thiết bị ngay sau rep baseline cuối — cùng thời điểm với neo N1.

    Hàm cũ giữ nguyên để v2 còn tái lập được.

    Returns
    -------
    dict : {'zc': float, 'ssc': float, 'wamp': float, 'sigma': float}
    """
    if len(rep_segments) < baseline_n:
        raise ValueError(f"cần ít nhất {baseline_n} rep segment, có {len(rep_segments)}")
    x = np.asarray(emg_filtered, dtype=float)
    base = np.concatenate([x[s:e] for s, e in rep_segments[:baseline_n]])
    sigma = float(np.std(base))
    return {
        "zc": k_zc * sigma,
        "ssc": (k_ssc * sigma) ** 2,
        "wamp": k_wamp * sigma,
        "sigma": sigma,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Trích xuất cho một đoạn
# ═══════════════════════════════════════════════════════════════════════════
def extract_extra_from_segment(
    seg, fs, thresholds, band=EMG_BAND, psd_method="fft", density=True
):
    """
    Tính 14 đại lượng mới cho MỘT đoạn tín hiệu (một "third" của rep).

    Parameters
    ----------
    seg : array-like
        Đoạn sEMG đã lọc bandpass.
    fs : float
    thresholds : dict
        Kết quả của `compute_trial_thresholds`.
    band, psd_method, density
        Xem `compute_psd`.

    Returns
    -------
    dict : tên đại lượng → giá trị float (có thể là NaN nếu đoạn quá ngắn)
    """
    seg = np.asarray(seg, dtype=float)
    freqs, pxx = compute_psd(seg, fs, band=band, psd_method=psd_method, density=density)

    return {
        # miền thời gian
        "MAV": mav(seg),
        "iEMG": iemg(seg, fs),
        "WL": waveform_length(seg),
        "ZC": zero_crossings(seg, fs, thresholds["zc"]),
        "SSC": slope_sign_changes(seg, fs, thresholds["ssc"]),
        "WAMP": willison_amplitude(seg, fs, thresholds["wamp"]),
        "DASDV": dasdv(seg),
        # miền tần số
        "PKF": peak_frequency(freqs, pxx),
        "LFHF": lf_hf_ratio(freqs, pxx),
        "FInsm5": finsm5(freqs, pxx),
        "SpecEnt": spectral_entropy(freqs, pxx),
        "SM1": spectral_moment(freqs, pxx, 1),
        "SM2": spectral_moment(freqs, pxx, 2),
        "SM3": spectral_moment(freqs, pxx, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Trích xuất theo rep — khớp 1:1 với notebook cell 11
# ═══════════════════════════════════════════════════════════════════════════
def extract_extra_per_rep(
    emg_filtered,
    time_arr,
    fs,
    rep_segments,
    subject,
    trial,
    thresholds=None,
    band=EMG_BAND,
    psd_method="fft",
    density=True,
):
    """
    Tính đặc trưng bổ sung cho từng rep, TÁI HIỆN CHÍNH XÁC vòng lặp của
    `extract_per_cycle_features` trong notebook cell 11.

    Phần "tái hiện chính xác" là thiết yếu: cùng điều kiện bỏ rep
    (len(seg) < fs·0.5, L < 16), cùng cách chia 3 phần (phần thứ ba nhận
    phần dư), và cùng cách đánh số `rep_idx = k+1` theo enumerate trên
    TOÀN BỘ rep_segments (kể cả rep bị bỏ, nên rep_idx có thể khuyết).
    Nhờ vậy bảng kết quả ghép được 1:1 với `per_cycle_features.csv` theo
    (subject, trial, rep_idx) mà không lệch hàng.

    Parameters
    ----------
    emg_filtered : array-like
        Tín hiệu toàn trial đã lọc bandpass.
    time_arr : array-like
        Trục thời gian tương ứng (giây).
    fs : float
    rep_segments : list[tuple[int, int]]
        Danh sách (start_idx, end_idx) từ `segment_reps` của notebook.
        CỐ Ý nhận từ ngoài vào thay vì tự phân đoạn lại, để không nhân bản
        `segment_reps` thành hai bản dễ lệch nhau.
    subject, trial : int
        Dùng làm khoá ghép.
    thresholds : dict or None
        Nếu None sẽ tự gọi `compute_trial_thresholds(emg_filtered)`.

    Returns
    -------
    pd.DataFrame
        Mỗi hàng một rep: subject, trial, rep_idx + 42 cột *_max/_min/_mean.
        Rỗng nếu không có rep nào hợp lệ.
    """
    emg_filtered = np.asarray(emg_filtered, dtype=float)
    n_total = len(rep_segments)
    if n_total == 0:
        return pd.DataFrame()

    if thresholds is None:
        thresholds = compute_trial_thresholds(emg_filtered)

    rows = []
    for k, (s, e) in enumerate(rep_segments):
        seg = emg_filtered[s:e]
        if len(seg) < int(fs * 0.5):
            continue

        third_len = len(seg) // 3
        if third_len < 16:
            continue
        thirds = [seg[:third_len], seg[third_len : 2 * third_len], seg[2 * third_len :]]

        # gom giá trị của 3 phần cho từng đại lượng rồi lấy max/min/mean
        collected = {name: [] for name in EXTRA_BASE_NAMES}
        for part in thirds:
            if len(part) < 16:
                continue
            feats = extract_extra_from_segment(
                part, fs, thresholds, band=band, psd_method=psd_method, density=density
            )
            for name, val in feats.items():
                if val is not None and np.isfinite(val):
                    collected[name].append(val)

        row = {}
        for name, vals in collected.items():
            if vals:
                row[f"{name}_max"] = float(np.max(vals))
                row[f"{name}_min"] = float(np.min(vals))
                row[f"{name}_mean"] = float(np.mean(vals))

        if not row:
            continue

        row.update({"subject": subject, "trial": trial, "rep_idx": k + 1})
        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Ghép & chuẩn hoá N1
# ═══════════════════════════════════════════════════════════════════════════
def merge_extra_features(per_cycle_df, extra_df, how="left", validate_coverage=True):
    """
    Ghép bảng đặc trưng bổ sung vào `per_cycle_features` theo
    (subject, trial, rep_idx).

    Dùng how='left' để giữ nguyên số hàng của bảng gốc — nếu một rep nào
    đó thiếu đặc trưng mới thì sẽ thành NaN và được báo ra, thay vì âm
    thầm làm mất hàng (điều sẽ phá vỡ việc so sánh với kết quả cũ).

    Returns
    -------
    pd.DataFrame
    """
    missing_keys = [k for k in MERGE_KEYS if k not in per_cycle_df.columns]
    if missing_keys:
        raise ValueError(f"per_cycle_df thiếu khoá ghép: {missing_keys}")
    if extra_df.empty:
        raise ValueError("extra_df rỗng — không có gì để ghép")

    out = per_cycle_df.merge(extra_df, on=MERGE_KEYS, how=how, suffixes=("", "_extra"))

    if len(out) != len(per_cycle_df):
        raise ValueError(
            f"Số hàng thay đổi sau merge ({len(per_cycle_df)} → {len(out)}) — "
            "khả năng extra_df có khoá (subject, trial, rep_idx) trùng lặp."
        )

    if validate_coverage:
        present = [c for c in EXTRA_BASE_COLS if c in out.columns]
        if present:
            n_missing = int(out[present].isna().all(axis=1).sum())
            pct = n_missing / len(out) * 100 if len(out) else 0.0
            print(
                f"[merge_extra_features] {len(out)} hàng, "
                f"{len(present)}/{len(EXTRA_BASE_COLS)} cột mới, "
                f"{n_missing} hàng ({pct:.1f}%) không có đặc trưng mới"
            )
    return out


def drop_collinear(df, cols, threshold=0.95, verbose=True):
    """
    Bước đầu của mục 5.4: loại đặc trưng có |Pearson| > threshold với một
    đặc trưng đã giữ trước đó.

    Giữ cột xuất hiện sớm hơn trong `cols` khi có cặp tương quan cao, nên
    thứ tự truyền vào có ý nghĩa — đặt các đặc trưng bạn tin cậy/dễ giải
    thích lên trước (ví dụ MNF, MDF trước SM1–SM3).

    Returns
    -------
    kept : list[str]
    dropped : list[tuple[str, str, float]]   (bị loại, vì trùng với, r)
    """
    present = [c for c in cols if c in df.columns]
    corr = df[present].corr().abs()

    kept, dropped = [], []
    for col in present:
        clash = None
        for k in kept:
            r = corr.loc[col, k]
            if np.isfinite(r) and r > threshold:
                clash = (col, k, float(r))
                break
        if clash is None:
            kept.append(col)
        else:
            dropped.append(clash)

    if verbose:
        print(f"[drop_collinear] giữ {len(kept)}/{len(present)} cột (|r| > {threshold})")
        for col, k, r in dropped:
            print(f"  bỏ {col:22s} (r={r:.3f} với {k})")
    return kept, dropped


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    # Console Windows mặc định cp1252 không in được tiếng Việt có dấu
    sys.stdout.reconfigure(encoding="utf-8")

    # ── Smoke test trên tín hiệu tổng hợp có "mỏi" mô phỏng ────────────────
    # Kiểm tra điều quan trọng nhất: các đặc trưng có ĐỔI CHIỀU ĐÚNG khi
    # phổ dịch về tần thấp và biên độ tăng (chính là dấu hiệu mỏi cơ).
    rng = np.random.default_rng(42)
    fs = 1259.0
    n_reps, rep_s = 20, 2.0

    sig, tim, segs = [], [], []
    cursor = 0
    for k in range(n_reps):
        n = int(rep_s * fs)
        t = np.arange(n) / fs
        frac = k / (n_reps - 1)
        centre = 120.0 - 50.0 * frac  # tần số trung tâm giảm dần → mỏi
        amp = 1.0 + 0.5 * frac  # biên độ tăng dần → mỏi
        burst = amp * np.sin(2 * np.pi * centre * t) * np.hanning(n)
        sig.append(burst + 0.05 * rng.standard_normal(n))
        tim.append(cursor / fs + t)
        segs.append((cursor, cursor + n))
        cursor += n

    sig = np.concatenate(sig)
    tim = np.concatenate(tim)

    thr = compute_trial_thresholds(sig)
    print(f"Ngưỡng theo trial: { {k: f'{v:.3e}' for k, v in thr.items()} }")

    out = extract_extra_per_rep(sig, tim, fs, segs, subject=1, trial=5, thresholds=thr)
    print(f"\nKết quả: {out.shape[0]} reps × {out.shape[1]} cột")
    assert out.shape[0] == n_reps, f"Mong {n_reps} reps, nhận {out.shape[0]}"

    have = [c for c in EXTRA_BASE_COLS if c in out.columns]
    print(f"Cột đặc trưng mới: {len(have)}/{len(EXTRA_BASE_COLS)}")
    assert len(have) == len(EXTRA_BASE_COLS), "Thiếu cột đặc trưng"
    assert out[have].isna().sum().sum() == 0, "Có NaN trong đặc trưng"

    print("\n--- Chiều biến thiên rep đầu → rep cuối (kỳ vọng khi mỏi) ---")
    expect = {
        "PKF_mean": "giảm",
        "LFHF_mean": "tăng",
        "FInsm5_mean": "tăng",
        "MAV_mean": "tăng",
        "WAMP_mean": "giảm",
        "SM1_mean": "tăng",
    }
    for col, want in expect.items():
        first, last = out[col].iloc[:3].mean(), out[col].iloc[-3:].mean()
        got = "tăng" if last > first else "giảm"
        flag = "OK " if got == want else "XEM LAI"
        print(f"  {flag} {col:14s} {first:12.4e} -> {last:12.4e}  ({got}, kỳ vọng {want})")

    # N1 normalization trên các cột mới
    out["rep_idx"] = out["rep_idx"].astype(int)
    norm = apply_N1_zscore(out, EXTRA_BASE_COLS, baseline_n=3)
    n1_cols = [f"{c}_N1" for c in EXTRA_BASE_COLS]
    print(f"\nSau N1: {len(n1_cols)} cột _N1, NaN = {norm[n1_cols].isna().sum().sum()}")

    # Kiểm tra bất biến theo thang PSD của các feature tỷ số
    seg_demo = sig[segs[0][0] : segs[0][1]]
    a = extract_extra_from_segment(seg_demo, fs, thr, density=True)
    b = extract_extra_from_segment(seg_demo, fs, thr, density=False)
    print("\n--- Bất biến với thang PSD (density=True vs False) ---")
    for name in ["PKF", "LFHF", "FInsm5", "SpecEnt"]:
        same = np.isclose(a[name], b[name], rtol=1e-9)
        print(f"  {'OK ' if same else 'KHAC'} {name}")
        assert same, f"{name} phải bất biến với thang PSD"
    for name in ["SM1", "SM2", "SM3"]:
        print(f"  (SMn phụ thuộc thang — đúng như thiết kế) {name}: "
              f"{a[name]:.3e} vs {b[name]:.3e}")

    # Welch branch
    w = extract_extra_from_segment(seg_demo, fs, thr, psd_method="welch")
    print(f"\nNhánh Welch chạy được: PKF={w['PKF']:.1f} Hz (fft: {a['PKF']:.1f} Hz)")

    print("\nSmoke test PASS")
