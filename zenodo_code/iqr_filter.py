# iqr_filter.py
import pandas as pd


def filter_outlier_trials(df, min_reps=10, verbose=True):
    """IQR Tukey fence per-muscle. Robust với n=13/muscle."""
    trial_n = (
        df.groupby(["subject", "trial"])[["N_total", "muscle"]].first().reset_index()
    )
    valid_indices = []
    removed_details = []

    for muscle, grp in trial_n.groupby("muscle"):
        Q1 = grp["N_total"].quantile(0.25)
        Q3 = grp["N_total"].quantile(0.75)
        IQR = Q3 - Q1
        lo = max(min_reps, Q1 - 1.5 * IQR)
        hi = Q3 + 1.5 * IQR

        valid = grp[(grp["N_total"] >= lo) & (grp["N_total"] <= hi)]
        removed = grp[~grp.index.isin(valid.index)]
        valid_indices.extend(list(zip(valid["subject"].values, valid["trial"].values)))

        if len(removed) > 0 and verbose:
            for _, r in removed.iterrows():
                removed_details.append(
                    f"  {muscle}: S{r['subject']} T{r['trial']} "
                    f"(N={r['N_total']}) [fence={lo:.0f}-{hi:.0f}]"
                )

    valid_set = set(valid_indices)
    mask = df.apply(lambda r: (r["subject"], r["trial"]) in valid_set, axis=1)
    if verbose:
        n_removed = len(df) - mask.sum()
        print(f"[IQR filter] {n_removed} reps removed ({len(df)} → {mask.sum()})")
        for d in removed_details:
            print(d)
    return df[mask].copy()


def load_filtered_and_enriched(csv_path, min_reps=10, verbose=False):
    """Nguồn duy nhất: load → IQR filter → build N1 feature sets."""
    from n1_features import build_all_feature_sets

    df_raw = pd.read_csv(csv_path)
    df_filtered = filter_outlier_trials(df_raw, min_reps=min_reps, verbose=verbose)
    feature_sets, df = build_all_feature_sets(df_filtered)
    return df_filtered, feature_sets, df
