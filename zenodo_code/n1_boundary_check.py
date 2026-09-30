"""
Kênh biên trong chính z-score N1 có rò rỉ rep_idx không?
=========================================================
Rolling slope đã bị phát hiện rò rỉ rep_idx qua cách xử lý biên cửa sổ. Câu
hỏi nghiêm trọng hơn: CHÍNH z-score N1 có kênh tương tự không? Nếu có thì nó
nằm trong đóng góp CỐT LÕI của bài, nên đây là phép kiểm đáng ưu tiên nhất.

CƠ CHẾ NGHI NGỜ CÓ DẠNG ĐẠI SỐ CHÍNH XÁC, không chỉ là phỏng đoán. Với 3 rep
baseline, z-score được tính bằng chính μ và σ của 3 rep đó, nên ba giá trị
z₁,z₂,z₃ bị ràng buộc:

    z₁+z₂+z₃ = 0        và        z₁²+z₂²+z₃² = n−1 = 2   (pandas std ddof=1)

Cực đại của |z| dưới hai ràng buộc này đạt khi hai giá trị bằng nhau:
(a,a,−2a) với 6a²=2 → |z|max = 2/√3 ≈ 1.1547.

Nghĩa là với rep_idx ≤ 3, MỌI cột z-score đều bị chặn trong [−1.1547, 1.1547],
ĐỒNG THỜI trên cả 54 cột. Với rep sau, z không bị chặn. Đó là một chữ ký "rep
≤ 3" mạnh hơn nhiều so với một cột slope đơn lẻ — và vì FCF = rep_idx/N_total,
nhận ra rep sớm là gần như biết FCF của nó.

PHÉP KIỂM (cùng công cụ đã dùng cho slope):
  control = xáo trộn THỨ TỰ HÀNG của toàn bộ 54 cột thô TRONG từng trial
  (cùng một phép hoán vị cho mọi cột, nên tương quan giữa các cột trong một
  hàng được giữ nguyên; chỉ quan hệ giữa đặc trưng và VỊ TRÍ rep bị phá).
  Sau đó tính lại neo N1 từ 3 hàng đầu của thứ tự mới.

  Nếu F1_ZSCORE vẫn thắng F0 trên dữ liệu đã xáo trộn → phần thắng đó đến từ
  chữ ký biên, không từ sinh lý.

Hai biến thể, mỗi biến thể có naive RIÊNG trên đúng tập hàng của nó:
  (I)  toàn bộ hàng
  (II) BỎ rep_idx ≤ 3 khỏi đánh giá (neo vẫn tính từ chúng) — cách duy nhất
       triệt tiêu chữ ký biên

Output: data/results/n1_boundary_check.csv

Usage:
    python n1_boundary_check.py
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
from iqr_filter import filter_outlier_trials
from n1_features import F0_COLS, apply_N1_zscore

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_CSV = os.path.join(BASE, "data", "processed", "per_cycle_features_v2.csv")
RESULTS_DIR = os.path.join(BASE, "data", "results")

TARGET = "FCF"
RF_KWARGS = {"n_estimators": 200, "max_depth": 10, "n_jobs": -1, "random_state": 42}
ALL_BASE = F0_COLS + EXTRA_BASE_COLS
BASELINE_N = 3
Z_BOUND = 2.0 / np.sqrt(3.0)  # 1.1547 — trần lý thuyết của |z| ở rep baseline
SEED = 42


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


def shuffle_rows_within_trial(df, cols, seed=SEED):
    """
    Xáo trộn THỨ TỰ HÀNG của block `cols` trong từng (subject, trial).

    Dùng CÙNG một phép hoán vị cho mọi cột để giữ nguyên tương quan giữa các
    đặc trưng trong một hàng; chỉ quan hệ đặc trưng ↔ vị trí rep bị phá. Nếu
    hoán vị riêng từng cột thì ta còn phá cả cấu trúc liên cột, tức thay đổi
    hai biến cùng lúc và phép kiểm mất tính cô lập.
    """
    rng = np.random.default_rng(seed)
    out = df.sort_values(["subject", "trial", "rep_idx"]).reset_index(drop=True)
    block = out[cols].to_numpy(copy=True)

    for _, idx in out.groupby(["subject", "trial"]).indices.items():
        idx = np.asarray(idx)
        block[idx] = block[idx[rng.permutation(len(idx))]]

    out[cols] = block
    return out


def verify_algebraic_bound(df, zcols):
    """Kiểm tra trần |z| ≤ 2/√3 ở rep baseline, và độ tách được của chữ ký."""
    print("\n--- Kiểm chứng ràng buộc đại số ---")
    base = df[df["rep_idx"] <= BASELINE_N]
    rest = df[df["rep_idx"] > BASELINE_N]

    max_base = float(np.nanmax(np.abs(base[zcols].to_numpy())))
    print(f"  max|z| ở rep_idx ≤ {BASELINE_N}: {max_base:.6f}  "
          f"(trần lý thuyết {Z_BOUND:.6f})")
    if max_base <= Z_BOUND + 1e-6:
        print("  -> Trần được tôn trọng: ràng buộc đại số xác nhận.")
    else:
        print("  -> Vượt trần: kiểm lại ddof của std hoặc baseline_n.")

    # Chữ ký: hàng nào có TOÀN BỘ |z| dưới trần
    def frac_under(sub):
        m = np.nanmax(np.abs(sub[zcols].to_numpy()), axis=1)
        return float(np.mean(m <= Z_BOUND + 1e-9))

    f_base, f_rest = frac_under(base), frac_under(rest)
    print(f"\n  Tỷ lệ hàng có MỌI |z| ≤ trần:")
    print(f"    rep_idx ≤ {BASELINE_N} : {f_base * 100:6.2f}%  (n={len(base)})")
    print(f"    rep_idx >  {BASELINE_N} : {f_rest * 100:6.2f}%  (n={len(rest)})")
    print(f"  -> Chữ ký tách được {f_base * 100:.1f}% vs {f_rest * 100:.1f}%: "
          f"{'RẤT MẠNH' if f_base - f_rest > 0.5 else 'yếu'}")
    print(f"  Rep baseline chiếm {len(base) / len(df) * 100:.2f}% tổng số hàng")
    return {"max_abs_z_baseline": max_base, "frac_under_baseline": f_base,
            "frac_under_rest": f_rest, "pct_rows_baseline": len(base) / len(df) * 100}


def evaluate(df, label, zcols, drop_baseline):
    """F0 vs F1_ZSCORE trên một dataframe, kèm naive cùng tập hàng."""
    d = df[df["rep_idx"] > BASELINE_N] if drop_baseline else df
    nv = naive_loso(d)["rmse"].mean()
    r0 = run_loso(RandomForestRegressor, [c for c in ALL_BASE if c in d.columns],
                  TARGET, d, model_kwargs=RF_KWARGS)
    r1 = run_loso(RandomForestRegressor, [c for c in zcols if c in d.columns],
                  TARGET, d, model_kwargs=RF_KWARGS)
    print(f"  {label:34s} n={len(d):6d}  naive={nv:6.2f}%  "
          f"F0={r0['rmse'].mean():6.2f}%  F1_ZSCORE={r1['rmse'].mean():6.2f}%  "
          f"lợi={r0['rmse'].mean() - r1['rmse'].mean():+6.2f}pp")
    return {"naive": nv, "f0": r0, "f1": r1,
            "gain": r0["rmse"].mean() - r1["rmse"].mean()}


def main():
    print("=" * 78)
    print("Kênh biên trong z-score N1: chữ ký 'rep ≤ 3' có rò rỉ rep_idx không?")
    print("=" * 78)

    df = pd.read_csv(V2_CSV)
    df = filter_outlier_trials(df, min_reps=10, verbose=False)

    # Tính lại N1 cho TOÀN BỘ 54 đại lượng bằng hàm canonical, để cả hai nhánh
    # (thật / xáo trộn) dùng đúng một công thức.
    zcols = [f"{c}_N1" for c in ALL_BASE]
    real = apply_N1_zscore(df.drop(columns=[c for c in zcols if c in df.columns]),
                           cols=ALL_BASE, baseline_n=BASELINE_N, report=False)
    print(f"\nDữ liệu: {len(real)} reps, "
          f"{real.groupby(['subject','trial']).ngroups} trials")

    diag = verify_algebraic_bound(real, zcols)

    # Control: xáo trộn thứ tự hàng của 54 cột thô, rồi tính lại neo
    shuf_raw = shuffle_rows_within_trial(df, ALL_BASE, seed=SEED)
    ctrl = apply_N1_zscore(shuf_raw.drop(columns=[c for c in zcols if c in shuf_raw.columns]),
                           cols=ALL_BASE, baseline_n=BASELINE_N, report=False)

    rows = []
    for variant, drop in [("(I) toàn bộ hàng", False),
                          ("(II) bỏ rep_idx ≤ 3", True)]:
        print(f"\n--- {variant} ---")
        a = evaluate(real, "dữ liệu THẬT", zcols, drop)
        b = evaluate(ctrl, "CONTROL xáo trộn trong trial", zcols, drop)
        true_gain = a["gain"] - b["gain"]

        print(f"  {'lợi THẬT (thật − control)':34s} {true_gain:+6.2f}pp")
        c_real = compare_paired(a["f0"], a["f1"], "F0-real", "F1Z-real")
        c_ctrl = compare_paired(b["f0"], b["f1"], "F0-ctrl", "F1Z-ctrl")

        rows.append({
            "variant": variant,
            "naive_real": a["naive"], "f0_real": a["f0"]["rmse"].mean(),
            "f1_real": a["f1"]["rmse"].mean(), "gain_real_pp": a["gain"],
            "p_real": c_real["p_value"],
            "naive_ctrl": b["naive"], "f0_ctrl": b["f0"]["rmse"].mean(),
            "f1_ctrl": b["f1"]["rmse"].mean(), "gain_control_pp": b["gain"],
            "p_control": c_ctrl["p_value"],
            "true_gain_pp": true_gain,
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "n1_boundary_check.csv"), index=False)

    # ── Phán quyết ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("PHÁN QUYẾT")
    print("=" * 78)

    for _, r in out.iterrows():
        leak = r["p_control"] < 0.05 and r["gain_control_pp"] > 0
        share = (r["gain_control_pp"] / r["gain_real_pp"] * 100
                 if r["gain_real_pp"] > 0 else np.nan)
        print(f"\n{r['variant']}:")
        print(f"  lợi thật = {r['gain_real_pp']:+.2f}pp (p={r['p_real']:.4f})")
        print(f"  lợi control = {r['gain_control_pp']:+.2f}pp (p={r['p_control']:.4f})"
              f"  -> {'CÓ rò rỉ biên' if leak else 'sạch'}")
        if np.isfinite(share):
            print(f"  phần artefact (cận trên) ≈ {share:.1f}% mức lợi")
        print(f"  lợi THẬT sau khi trừ control = {r['true_gain_pp']:+.2f}pp")

    clean = out[out["variant"].str.startswith("(II)")].iloc[0]
    full = out[out["variant"].str.startswith("(I)")].iloc[0]
    print("\n" + "-" * 78)
    # Rò rỉ đòi hỏi control lợi theo chiều DƯƠNG. Nếu chỉ kiểm p mà bỏ dấu thì
    # một control lợi ÂM có ý nghĩa (N1 còn kém hơn F0 trên dữ liệu xáo trộn —
    # tức bằng chứng KHÔNG rò rỉ) lại bị kết luận thành "có vấn đề".
    control_leaks = clean["p_control"] < 0.05 and clean["gain_control_pp"] > 0
    if not control_leaks:
        direction = ("âm" if clean["gain_control_pp"] < 0 else "không có ý nghĩa")
        print("Ở biến thể (II) — bỏ rep baseline khỏi đánh giá — control KHÔNG còn rò rỉ")
        print(f"(lợi control = {clean['gain_control_pp']:+.2f}pp, {direction}), "
              f"và N1 vẫn lợi {clean['gain_real_pp']:+.2f}pp (p={clean['p_real']:.4f}).")
        print("=> Đóng góp của N1 KHÔNG phải artefact chữ ký biên. Kênh biên có tồn tại")
        print(f"   về mặt đại số nhưng chỉ ảnh hưởng {diag['pct_rows_baseline']:.1f}% số hàng,")
        print("   và loại các hàng đó đi thì kết luận N1 vẫn giữ nguyên.")
    else:
        print(f"Ở biến thể (II), control VẪN có ý nghĩa (p={clean['p_control']:.4f}).")
        print("=> Cần điều tra thêm: chữ ký biên không phải nguồn duy nhất,")
        print("   hoặc phép xáo trộn chưa phá hết cấu trúc liên quan tới vị trí.")

    print(f"\nLưu ý khi viết bài: phần trừ đi là ƯỚC LƯỢNG CẬN TRÊN của artefact,")
    print("không phải giá trị đo chính xác — phép trừ giả định artefact cộng tính")
    print("và bằng nhau ở hai nhánh.")
    print(f"\nBiến thể (I) lợi {full['gain_real_pp']:+.2f}pp, "
          f"(II) lợi {clean['gain_real_pp']:+.2f}pp trên "
          f"{100 - diag['pct_rows_baseline']:.1f}% hàng còn lại.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
