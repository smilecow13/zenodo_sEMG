"""
Statistical Utilities Module
==============================
Bộ công cụ thống kê nâng cao theo yêu cầu thầy hướng dẫn.

Bổ sung so với hiện tại (chỉ có Wilcoxon):
  - Bootstrap CI (10,000 lần)
  - Friedman + Nemenyi post-hoc (so sánh >2 models)
  - Holm-Bonferroni correction
  - Bland-Altman plot (cho RIR agreement)
  - Wilson score CI (cho tỷ lệ classification)

Usage:
    from stats_utils import bootstrap_ci, friedman_nemenyi, bland_altman_plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap Confidence Interval
# ══════════════════════════════════════════════════════════════════════════════
def bootstrap_ci(scores, n_boot=10000, alpha=0.05, statistic=np.mean, random_state=42):
    """
    Bootstrap confidence interval cho một metric.

    Parameters
    ----------
    scores : array-like
        Giá trị metric qua các fold (e.g., RMSE per subject)
    n_boot : int
        Số lần bootstrap (thầy yêu cầu 10,000)
    alpha : float
        Significance level (default 0.05 → 95% CI)
    statistic : callable
        Hàm thống kê (default: np.mean)
    random_state : int

    Returns
    -------
    dict with point_estimate, ci_lower, ci_upper, ci_width
    """
    rng = np.random.RandomState(random_state)
    scores = np.asarray(scores)
    n = len(scores)

    boot_stats = np.array(
        [statistic(rng.choice(scores, size=n, replace=True)) for _ in range(n_boot)]
    )

    ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
    ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return {
        "point_estimate": statistic(scores),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
        "n_boot": n_boot,
        "alpha": alpha,
    }


def bootstrap_ci_table(results_dict, metric="rmse", n_boot=10000, alpha=0.05):
    """
    Bootstrap CI cho nhiều configurations.

    Parameters
    ----------
    results_dict : dict
        {config_name: pd.DataFrame with metric column}
    metric : str

    Returns
    -------
    pd.DataFrame
        Summary table với CI cho mỗi config
    """
    rows = []
    for name, res_df in results_dict.items():
        ci = bootstrap_ci(res_df[metric].values, n_boot=n_boot, alpha=alpha)
        rows.append(
            {
                "config": name,
                f"{metric}_mean": ci["point_estimate"],
                f"{metric}_ci_lower": ci["ci_lower"],
                f"{metric}_ci_upper": ci["ci_upper"],
                f"{metric}_ci_width": ci["ci_width"],
            }
        )
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Friedman Test + Nemenyi Post-Hoc
# ══════════════════════════════════════════════════════════════════════════════
def friedman_nemenyi(results_dict, metric="rmse", alpha=0.05):
    """
    Friedman test + Nemenyi post-hoc cho so sánh >2 models.

    Thầy yêu cầu: KHÔNG dùng t-test lặp lại nhiều lần.
    Friedman là non-parametric alternative cho repeated-measures ANOVA.

    Parameters
    ----------
    results_dict : dict
        {model_name: pd.DataFrame with metric column and 'test_subject'}
    metric : str
    alpha : float

    Returns
    -------
    dict with:
        friedman_stat, friedman_p, significant, posthoc_df (Nemenyi p-values)
    """
    try:
        import scikit_posthocs as sp
    except ImportError:
        print("⚠️ pip install scikit-posthocs required for Nemenyi post-hoc")
        sp = None

    # Build matrix: rows = subjects, cols = models
    model_names = list(results_dict.keys())
    # Merge all on test_subject
    merge_key = "test_subject"
    all_dfs = []
    for name, res in results_dict.items():
        df_temp = res[[merge_key, metric]].rename(columns={metric: name})
        all_dfs.append(df_temp)

    merged = all_dfs[0]
    for df_temp in all_dfs[1:]:
        merged = merged.merge(df_temp, on=merge_key)

    matrix = merged[model_names].values  # (n_subjects, n_models)

    # Friedman test
    stat, p_value = sp_stats.friedmanchisquare(
        *[matrix[:, i] for i in range(len(model_names))]
    )

    result = {
        "friedman_stat": stat,
        "friedman_p": p_value,
        "significant": p_value < alpha,
        "n_models": len(model_names),
        "n_subjects": matrix.shape[0],
    }

    print(
        f"Friedman χ²={stat:.4f}, p={p_value:.4f} "
        f"{'[SIG]' if p_value < alpha else '[n.s.]'}"
    )

    # Nemenyi post-hoc (only if Friedman is significant)
    if sp is not None and p_value < alpha:
        posthoc = sp.posthoc_nemenyi_friedman(matrix)
        posthoc.index = model_names
        posthoc.columns = model_names
        result["posthoc_df"] = posthoc
        print(f"\nNemenyi post-hoc p-values:")
        print(posthoc.round(4))
    else:
        result["posthoc_df"] = None
        if p_value >= alpha:
            print("  Friedman n.s. → Nemenyi post-hoc not performed")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Holm-Bonferroni Correction
# ══════════════════════════════════════════════════════════════════════════════
def holm_bonferroni(p_values, alpha=0.05):
    """
    Holm-Bonferroni correction cho multiple comparisons.

    Parameters
    ----------
    p_values : dict
        {comparison_name: p_value}
    alpha : float

    Returns
    -------
    pd.DataFrame with columns: comparison, p_value, p_adjusted, significant
    """
    names = list(p_values.keys())
    pvals = [p_values[n] for n in names]
    n = len(pvals)

    # Sort by p-value
    order = np.argsort(pvals)
    sorted_names = [names[i] for i in order]
    sorted_pvals = [pvals[i] for i in order]

    # Holm là thủ tục STEP-DOWN: gặp test đầu tiên không vượt ngưỡng thì DỪNG,
    # mọi test có p lớn hơn đều không được bác bỏ. Nếu kiểm từng dòng độc lập
    # (thiếu quy tắc dừng) thì ví dụ p=[0.03, 0.04] với n=2 sẽ cho 0.03 fail
    # (ngưỡng 0.025) nhưng 0.04 lại "significant" (ngưỡng 0.05) — vô lý.
    rows = []
    stopped = False
    running_max = 0.0
    for rank, (name, pval) in enumerate(zip(sorted_names, sorted_pvals)):
        adjusted_alpha = alpha / (n - rank)
        if not stopped and pval >= adjusted_alpha:
            stopped = True
        # p điều chỉnh, ép đơn điệu không giảm theo thứ hạng
        running_max = max(running_max, min(1.0, (n - rank) * pval))
        rows.append(
            {
                "comparison": name,
                "p_value": pval,
                "adjusted_alpha": adjusted_alpha,
                "p_adjusted": running_max,
                "significant": not stopped,
            }
        )

    result = pd.DataFrame(rows)
    print("Holm-Bonferroni correction:")
    print(result.to_string(index=False))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Bland-Altman Plot
# ══════════════════════════════════════════════════════════════════════════════
def bland_altman_plot(actual, predicted, title="Bland-Altman: RIR", save_path=None):
    """
    Bland-Altman plot cho đánh giá agreement RIR predicted vs actual.

    Thầy yêu cầu cho N2 (dual-head RIR prediction).

    Parameters
    ----------
    actual, predicted : array-like
    title : str
    save_path : str or None

    Returns
    -------
    dict with mean_diff, std_diff, upper_loa, lower_loa
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mean_vals = (actual + predicted) / 2
    diff_vals = actual - predicted

    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals, ddof=1)
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(mean_vals, diff_vals, alpha=0.4, s=10, color="steelblue")
    ax.axhline(
        mean_diff, color="red", linestyle="-", label=f"Mean diff: {mean_diff:.2f}"
    )
    ax.axhline(
        upper_loa, color="gray", linestyle="--", label=f"+1.96 SD: {upper_loa:.2f}"
    )
    ax.axhline(
        lower_loa, color="gray", linestyle="--", label=f"-1.96 SD: {lower_loa:.2f}"
    )
    ax.set_xlabel("Mean of Actual and Predicted (reps)")
    ax.set_ylabel("Difference (Actual − Predicted) (reps)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Bland-Altman plot saved to {save_path}")
    plt.close()

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "upper_loa": upper_loa,
        "lower_loa": lower_loa,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Wilson Score Confidence Interval
# ══════════════════════════════════════════════════════════════════════════════
def wilson_score_ci(n_success, n_total, alpha=0.05):
    """
    Wilson score CI cho tỷ lệ phân loại đúng.
    Dùng cho binary classification ablation (nếu giữ).

    Parameters
    ----------
    n_success : int
    n_total : int
    alpha : float

    Returns
    -------
    dict with p_hat, ci_lower, ci_upper
    """
    z = sp_stats.norm.ppf(1 - alpha / 2)
    p_hat = n_success / n_total

    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total) / denom

    return {
        "p_hat": p_hat,
        "ci_lower": center - margin,
        "ci_upper": center + margin,
    }


if __name__ == "__main__":
    # Quick smoke test
    np.random.seed(42)
    scores = np.random.normal(25, 3, 13)  # Fake RMSE from 13 LOSO folds
    ci = bootstrap_ci(scores)
    print(
        f"Bootstrap CI: {ci['point_estimate']:.2f} "
        f"[{ci['ci_lower']:.2f}, {ci['ci_upper']:.2f}]"
    )

    # Bland-Altman
    actual = np.random.randint(0, 11, 100).astype(float)
    predicted = actual + np.random.normal(0, 1.5, 100)
    ba = bland_altman_plot(actual, predicted, title="Test BA", save_path=None)
    print(
        f"Bland-Altman: mean_diff={ba['mean_diff']:.2f}, "
        f"LoA=[{ba['lower_loa']:.2f}, {ba['upper_loa']:.2f}]"
    )

    # Wilson score
    ws = wilson_score_ci(85, 100)
    print(f"Wilson CI: {ws['p_hat']:.2f} [{ws['ci_lower']:.3f}, {ws['ci_upper']:.3f}]")
