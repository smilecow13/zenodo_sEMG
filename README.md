# sEMG_IRL_test — đo thử sEMG bicep curl trên người thật bằng pipeline Zenodo

Công cụ cho nhóm sEMG/FCF: kiểm chất lượng bản ghi ADS1292R, cắt rep, và so quy luật mỏi trên người
thật với 26 trial bicep của dataset Zenodo 14182446.

**Nguồn:** mọi code trong `zenodo_code/` và dữ liệu trong `data/` được chép nguyên văn từ repo chính
`sEMG_FCF_Wearable`, commit `3f64e1bad68f0c09705d0c4cb4d8cd2a1c0510dc`. `SOURCES.sha256` ghi checksum và
đường dẫn gốc của từng file. **Không sửa các file đó ở đây.** Nếu cần sửa, sửa ở repo chính, chép lại
và cập nhật bảng. Mọi con số khoa học (K1–K5, decision note) nằm ở repo chính.

**Repo private.** Để private tới lúc nộp bài: push lên public là không rút lại được.

**Mục đích:** kiểm xem mạch ADS1292R cộng pipeline Zenodo có thấy quy luật mỏi trên người thật
không. Cách kiểm: ρ Spearman của tần số trung vị (MDF) theo thời gian, đặt cạnh phân bố của 26 trial
bicep trên Zenodo.

**Repo này KHÔNG trả lời:** mô hình đoán FCF có tốt không, hay sEMG có thắng bộ đếm rep không. Dữ liệu
chỉ có một người, nên không có phép chấm nào hợp lệ.

**Điều kiện đo:** chỉ thu sEMG, không có IMU. Bài tập là bicep curl với một vật bất kỳ, không theo
%1RM, tập theo kiểu sức bền (endurance). Giống Zenodo ở ba điểm: chỉ có sEMG, mỗi trial một cơ đích,
và không có thông tin tải.

## Cài đặt

```
git lfs install                       # một lần trên mỗi máy
git clone <url repo>                  # file mô hình 82 MB đi qua Git LFS
cd sEMG_IRL_test
pip install -r requirements.txt       # Python 3.14; phiên bản được ghim, xem chú thích trong file
python verify_copies.py               # mọi bản sao và file mô hình phải OK
```

Nếu `verify_copies.py` báo `models/g12_v2b_fold_models.joblib` lệch và file chỉ nặng vài trăm byte,
tức là mới tải con trỏ LFS về: chạy `git lfs pull`.

**Dữ liệu thô Zenodo (không bắt buộc).** Chỉ cần cho các self-test và cho `build_reference.py
--check-raw / --mains`. Tải dataset Zenodo 14182446 (link: ____) rồi đặt thư mục `sEMG_data` vào
`dataset/sEMG_data/`, hoặc trỏ biến môi trường `ZENODO_SEMG_DIR` tới nó. Thư mục `dataset/` bị git
bỏ qua. Không có dữ liệu thô thì mọi công cụ chạy trên bản ghi tự thu vẫn dùng được, vì các bảng tham
chiếu đã được tính sẵn trong `reference/`.

## Nội dung

| đường dẫn | là gì |
|---|---|
| `qc_recording.py` | kiểm chất lượng bản ghi: mẫu rơi, fs thật, độ dài, bão hoà, lead-off, nhiễu điện lưới, hai kênh |
| `analyze_recording.py` | cắt rep (`--reps-only`), ρ của MDF theo cửa sổ (W) và theo rep (R); `--self-test` |
| `model_direction.py` | kiểm **chiều** của 13 mô hình fold G12 trên một set tự thu; `--train`, `--self-test` |
| `zenodo_windowed_mdf.py` | MDF/MNF theo cửa sổ 4 s, bước 2 s, toàn phổ (logic của `dataset/code.ipynb` cell 1, repo chính) |
| `build_reference.py` | sinh `reference/` từ `data/`; `--check-raw` tái lập ρ từ dữ liệu thô; `--mains` sinh mốc điện lưới |
| `verify_copies.py` | kiểm `SOURCES.sha256` |
| `zenodo_code/` | 11 module pipeline Zenodo, bản sao của `preprocessing/` ở repo chính |
| `data/processed/` | `per_cycle_features.csv` (v1), `per_cycle_features_v2b.csv`, `trial_summary_with_flags.csv` |
| `data/results/` | `g12_rowlevel_predictions_v2b.csv`: dự đoán out-of-fold của G12, mốc cho `--train` |
| `reference/` | bảng tham chiếu bicep Zenodo, sinh bằng `build_reference.py` |
| `models/` | 13 mô hình fold (Git LFS), sinh bằng `model_direction.py --train` |
| `recordings/` | dữ liệu tự thu. **Bị git bỏ qua** cho tới khi hội đồng đạo đức trả lời câu hỏi về self-test; chia sẻ qua Drive |

## Yêu cầu khi thu

**Cấu hình ADS1292R**
- **2000 SPS, không dùng 1000.** Theo datasheet, bộ lọc decimation sinc3 của ADS1292R có dải thông
  −3 dB ≈ 0.262 × tốc độ dữ liệu, tức khoảng 262 Hz ở 1000 SPS và khoảng 524 Hz ở 2000 SPS. Ở 1000 SPS
  thì phần trên của dải 20–450 Hz bị cắt, và MDF bị kéo xuống ngay từ lúc thu. Kiểm lại con số này
  trên datasheet đúng phiên bản chip.
- **Gain 6.** Chip không chặn thành phần DC. Điện áp lệch của điện cực có thể tới vài trăm mV, trong
  khi dải vào ở gain 6 là ±VREF/6 ≈ ±0.40 V và ở gain 12 chỉ còn ±0.20 V.
- **Một kênh:** kênh 1 đặt trên bicep, kênh 2 để trống.
- Toàn hệ chạy pin, **kể cả laptop: rút sạc khi đang ghi.** Không nối bất cứ thứ gì với điện lưới khi
  điện cực đang dán.

**Định dạng file** (mỗi set một file, đặt trong `recordings/`)
- Firmware phải ghi **mọi mẫu** ở đúng fs, mỗi mẫu một dòng, có dòng tiêu đề. **Không** ghi bằng log
  Serial Monitor kiểu `Raw: … | Env: …`: log đó chỉ in một mẫu mỗi lần làm mới màn hình, tức lấy
  mẫu thưa mà không lọc trước, nên phổ bị chồng lấn (aliasing) và MDF mất nghĩa.
- Định dạng firmware hiện tại dùng được trực tiếp:
  `sample_idx,timestamp_us,ch1_raw24,ch2_raw24,raw_emg16,filtered_emg16,lead_off`.
  `sample_idx` giúp phát hiện mẫu rơi. Dùng `ch1_raw24` (mã ADC 24-bit chưa lọc), vì pipeline tự lọc.
- Đặt tên theo mẫu `YYYYMMDD_setN_bicepR.csv`.

**Quy trình một set**
1. Dán điện cực theo SENIAM, chuẩn bị da kỹ, cố định dây vào tay bằng băng dính y tế.
2. Bấm ghi sEMG và bấm quay video. **Đánh dấu đồng bộ:** nắm chặt tay mạnh 3 lần thật nhanh, để thấy
   được cả trên video lẫn trên tín hiệu. Sau đó nghỉ khoảng 10 s (đoạn này dùng để xem nhiễu nền).
3. Tập theo metronome cho tới điểm kết thúc đã khai. Ghi thêm khoảng 5 s sau rep cuối rồi mới dừng.
4. Từ video, ghi vào sổ: giây bắt đầu rep đầu tiên và giây kết thúc rep cuối (tính theo đồng hồ của
   bản ghi sEMG, căn theo mốc đồng bộ), số rep thật, và mức mỏi tự đánh giá.
5. Set phải đủ dài. Dưới khoảng 60 s thì ít hơn 30 cửa sổ, và |ρ| phải vượt 0.36 mới phân biệt được
   với 0. Miền sức bền như Zenodo là khoảng 3–5 phút.

**Nhiễu 50 Hz.** Pipeline Zenodo không có bộ lọc notch, và 50 Hz nằm trong dải 20–450 Hz. **Không tự
thêm notch**, vì như vậy là đổi pipeline và không còn so trực tiếp với Zenodo. Xử lý ở phía phần cứng,
và kiểm bằng mục [6] của `qc_recording.py`: trên Zenodo bicep, phần công suất nằm ở các vạch 50·k Hz
chỉ 3.5–4.4%, đúng bằng mức của một phổ không có nhiễu điện lưới.

## Các bước chạy

```
# 1. Một lần: bản sao khớp nguồn; (có dữ liệu thô) script khớp Zenodo
python verify_copies.py
python build_reference.py --check-raw          # cần dữ liệu thô
python analyze_recording.py --self-test        # cần dữ liệu thô
python model_direction.py --self-test          # cần dữ liệu thô

# 2. Điền và ghi ngày vào mục "Khai trước" ở cuối file này, TRƯỚC khi mở bản ghi

# 3. Mỗi set — kiểm chất lượng TRƯỚC
python qc_recording.py recordings/20261005_set1_bicepR.csv --fs 2000 --col ch1_raw24 --col2 ch2_raw24

# chỉ phân đoạn rep
python analyze_recording.py recordings/20261005_set1_bicepR.csv --fs 2000 --col ch1_raw24 \
    --time-col timestamp_us --start 14.2 --end 262.8 --n-video 91 --reps-only

# phân tích đầy đủ, và kiểm chiều của mô hình
python analyze_recording.py recordings/20261005_set1_bicepR.csv --fs 2000 --col ch1_raw24 \
    --time-col timestamp_us --start 14.2 --end 262.8 --n-video 91
python model_direction.py recordings/20261005_set1_bicepR.csv --fs 2000 --col ch1_raw24 \
    --time-col timestamp_us --start 14.2 --end 262.8
```

`--start 14.2 --end 262.8 --n-video 91` chỉ là số ví dụ. Thay bằng mốc và số rep lấy từ video của
chính set đó.

- **`--reps-only`** chỉ ghi `rep_boundaries.csv` và `segmentation.png`, và in bảng rep ra terminal.
  `rep_boundaries.csv` có một dòng cho **mỗi đoạn giữa hai biên**: `status = rep` hoặc `dropped` (đoạn
  bị bỏ vì dài quá 4.5 s hoặc ngắn hơn 1.2 s, thường là hai rep bị gộp). Các cột theo quy ước
  `wsd_rep_boundaries.csv` của repo chính:
  `rep_idx, status, rep_start_s, rep_end_s, rep_duration_s, start_sample, end_sample, N_total, pipeline`.
  Thời gian tính theo **đồng hồ của bản ghi gốc**, không tính từ `--start`, nên so thẳng được với
  video. Biên đầu của rep đầu tiên và biên cuối của rep cuối cùng chính là `--start` và `--end`.
- `--time-col`: đơn vị đọc theo hậu tố tên cột (`_us`, `_ms`, còn lại là giây). Khi có cột này,
  script ước lượng fs thật và dừng nếu lệch hơn 1%.
- Thiếu `--start` / `--end` thì script vẫn chạy nhưng sẽ cảnh báo, vì đoạn nghỉ ở đầu và cuối làm lệch
  ngưỡng cắt rep.
- Kết quả được ghi vào `recordings/<tên file>_out/`, thư mục này cũng bị git bỏ qua. **Xem
  `diagnostic.png` trước khi đọc số.**

Self-test trên hai trial Zenodo (S1/T5 và S3/T6) cho ρ, N và MDF từng rep khớp tuyệt đối với bảng
tham chiếu và v2b. Khi nâng chính các trial đó lên 2000 Hz rồi cho đi qua đường resample, ρ lệch tối
đa 0.0015 và N không đổi.

## Kiểm chiều của mô hình (`model_direction.py`)

**Câu hỏi:** FCF mà mô hình Zenodo đoán có tăng dần theo số rep trên set của mình không? Phép này
**không chấm độ chính xác** và script không in RMSE. Lý do: trên chính Zenodo, RMSE của từng trial
bicep đã trải từ 13.5% đến 29.5% (G12, v2b, GT2), nên RMSE của một set không phân biệt được mạch tốt,
mô hình tốt hay may.

**Mô hình:** 13 mô hình fold của cấu hình headline Zenodo (v2b, GT2, 12 cơ, mRMR theo nhóm in-fold
k = 12, RF 200/10/42), dựng lại đúng như `preprocessing/decompose_bicep_p2.py` ở repo chính. Mỗi mô
hình train trên 12 người, với danh sách cột chọn trong chính fold đó. Người tự thu không nằm trong tập
train của mô hình nào, nên cả 13 mô hình đều ở đúng tình huống đã sinh phân bố tham chiếu (dự đoán
out-of-fold). Không có danh sách chọn toàn cục, nên không có chọn lọc ngoài vòng CV.

Mỗi mô hình khoảng 18 MB trong bộ nhớ (268 000 nút). File trong `models/` là 13 mô hình nén, 82 MB.
Muốn dựng lại: `python model_direction.py --train`, khoảng 1 phút; lệnh này assert khớp tuyệt đối
10 318 dự đoán trong `data/results/g12_rowlevel_predictions_v2b.csv`.

**Đặc trưng:** đủ 54 đại lượng v2b, chuẩn hoá N1 (z, ratio, diff) neo theo 3 rep đầu của chính set,
rồi bỏ rep ≤ 3 (GT2).

**Phát hiện phụ khi làm self-test.** 12 cột F0 trong v2/v2b **không** bit-identical với v1 khi đọc
bằng `round_trip`. Chúng là giá trị v1 đọc bằng parser mặc định, vì `build_features_v2.py` (repo
chính) đọc v1 bằng parser mặc định rồi ghi ra. Độ lệch lớn nhất ở TP/RMS, khoảng 1e-12 tương đối. Đặc
trưng trích lại khớp bit-identical với **v1**. Self-test assert cả chuỗi này. Mọi kết quả đã có vẫn
nhất quán trong v2b.

**Tham chiếu:** `reference/zenodo_bicep_pred_rho.csv`, ρ(FCF đoán, rep_idx) trên 24 trial bicep
(2 trial đã bị lọc IQR loại):

| p10 | p25 | p50 | p75 | p90 |
|---|---|---|---|---|
| +0.50 | +0.61 | +0.77 | +0.86 | +0.91 |

**Đọc kết quả:** con số chính là ρ trung vị của 13 mô hình. Script cũng in dải min–max, số mô hình cho
ρ > 0, vị trí trong phân bố trên, và báo khi N của set nằm ngoài dải N_total bicep có trong tập train
[63, 156].

Ví dụ vì sao chỉ được dùng mô hình chưa thấy người đó: chạy trên trial S1/T5 của Zenodo, vốn nằm trong
tập train của 12/13 mô hình, ρ trung vị ra +0.959, cao hơn cả 24 trial tham chiếu. Mô hình duy nhất
chưa thấy S1 chỉ cho +0.825.

## Hai loại ρ, hai pipeline, không được đặt lẫn

| | (W) ρ theo cửa sổ | (R) ρ theo rep |
|---|---|---|
| đơn vị | cửa sổ 4 s, bước 2 s | một rep (trung bình của 3 phần rep) |
| phổ | toàn phổ dương từ 0 đến fs/2 | dải 20–450 Hz |
| trục x | thời gian | `rep_idx` |
| nguồn Zenodo | `trial_summary_with_flags.csv` | `per_cycle_features_v2b.csv` |
| có cần cắt rep không | **không** | có |

Repo chính có **hai** biến thể MDF theo cửa sổ 4 s/2 s và chúng cho kết quả khác nhau.
`trial_summary_with_flags.csv` tính trên **toàn phổ**, còn `quality_metrics_v2.csv` tính trên **dải
20–450 Hz**. Ví dụ S1/trial 5 cho ρ = −0.93558 và −0.93595. `zenodo_windowed_mdf.py` là biến thể toàn
phổ, và `--check-raw` xác nhận nó tái lập đúng bảng tham chiếu.

**(W) là phép kiểm chính**, vì nó không phụ thuộc vào bộ cắt rep. (R) chỉ có nghĩa khi số rep mà bộ cắt
tìm được khớp với số rep đếm từ video.

## Tham chiếu Zenodo bicep (26 trial)

| | p10 | p25 | p50 | p75 | p90 |
|---|---|---|---|---|---|
| (W) ρ_MDF | −0.911 | −0.833 | −0.655 | −0.371 | −0.191 |
| (R) ρ_MDF | −0.863 | −0.808 | −0.562 | −0.380 | −0.196 |
| N_total (theo bộ cắt rep) | 66 | 78 | 90 | 111 | 144 |
| thời lượng rep trung vị (s) | 1.98 | 2.10 | 2.54 | 3.04 | 3.73 |
| thời lượng set (s) | 180 | 208 | 262 | 323 | 341 |

Có 2/26 trial cho ρ_MDF ≥ 0 (S3 và S7, trial 5), và cả hai được gắn `is_outlier`. Nghĩa là ngay trên
dataset gốc cũng có khoảng 8% trial không thấy MDF giảm.

Metadata Zenodo không ghi khối lượng tạ. Mốc duy nhất để chọn vật nặng là **hình dạng set**: khoảng
3–5 phút, khoảng 70–110 rep.

### ρ = −0.3 có nghĩa gì còn tuỳ vào độ dài set

Ngưỡng |ρ| có ý nghĩa ở mức 5% (hai phía), dưới giả thuyết không có xu hướng:

| số cửa sổ (W) hoặc số rep (R) | 20 | 30 | 45 | 60 | 90 | 130 |
|---|---|---|---|---|---|---|
| \|ρ\| tới hạn | 0.44 | 0.36 | 0.29 | 0.25 | 0.21 | 0.17 |

## Chỗ giống và chỗ khác Zenodo

**Tần số lấy mẫu.** Zenodo lấy mẫu ở 1259 Hz. Có những chỗ **phụ thuộc fs một cách ngầm**: WL, DASDV
và ngưỡng SSC/WAMP tính trên hiệu hai mẫu liên tiếp; (W) lấy toàn phổ tới fs/2; ngưỡng `L < 16` mẫu.
Vì vậy mọi script đều **resample về 1259 Hz** bằng `resample_poly` (có lọc chống aliasing), dựng lại
trục thời gian, rồi lọc Butterworth bậc 4, 20–450 Hz bằng `filtfilt` như Zenodo.

**Cắt rep.** `min_rep_s = 1.2`, `max_rep_s = 4.5`. Rep ngoài khoảng này **bị bỏ mà không báo lỗi**.
Với `--reps-only`, các đoạn bị bỏ hiện ra trong `rep_boundaries.csv` với `status = dropped`. Chọn nhịp
gần trung vị Zenodo (khoảng 2.5 s mỗi rep) để còn khoảng dư khi mỏi làm chậm. `prominence =
0.5·std(đường bao)` tính trên toàn bản ghi, nên phải cắt bản ghi theo video (`--start/--end`).

**Đáp án.** Không có IMU, nên video là nguồn duy nhất cho N_total và thời điểm dừng.

**Biên độ.** Đơn vị và gain khác Zenodo. ρ của MDF và các đặc trưng đã chuẩn hoá N1 không đổi khi đổi
thang biên độ.

## Khai trước — điền và ghi ngày TRƯỚC khi mở bản ghi đầu tiên

Chưa khoá. Các con số bên dưới chỉ là đề xuất.

- Ngày khoá: ____
- Vật nặng: ____ kg (cân và ghi lại; các buổi sau dùng lại đúng vật này)
- Nhịp: metronome ____ s/rep
- Điểm kết thúc: ☐ thang mỏi tự đánh giá, dừng ở mức 2 (giống Zenodo) ☐ thất bại kỹ thuật
- Tay: ☐ phải ☐ trái. Điện cực theo SENIAM; RLD đặt ở ____
- fs thu: ____ SPS
- Điều kiện chất lượng (`qc_recording.py`): độ dài ≥ ____ s (đề xuất 60), nhiễu điện lưới ≤ ____ %
  (đề xuất 10).
- Phép kiểm chính: (W) ρ_MDF theo cửa sổ 4 s/2 s, toàn phổ.
- **Kỳ vọng:** ρ_MDF nằm trong khoảng p10–p90 của Zenodo bicep, tức [−0.91, −0.19].
- **"Thấy"** khi: ρ_MDF < 0 **và** |ρ_MDF| vượt ngưỡng tới hạn ứng với số cửa sổ thật của set.
- **"Không thấy"** khi: ρ_MDF ≥ 0, hoặc |ρ_MDF| không vượt ngưỡng tới hạn. 2/26 trial Zenodo cũng rơi
  vào trường hợp này, nên một set "không thấy" chưa đủ để kết luận mạch hỏng.
- Phép kiểm phụ (R) chỉ được chạy khi |N_bộ cắt − N_video| ≤ ____ rep.
- Kiểm chiều của mô hình (P), ghi rõ là kiểm chiều, không phải độ chính xác:
  - **kỳ vọng** ρ trung vị 13 mô hình nằm trong [+0.50, +0.91];
  - **"đúng chiều"** khi ρ trung vị > 0 **và** vượt ngưỡng tới hạn ứng với số hàng GT2, **và** ρ > 0 ở
    ít nhất ____/13 mô hình.
  - Chỉ đọc (P) khi (R) hợp lệ.
