from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Bins : [1–6], [7–12], [13+]
BINS: list[tuple[int, int]] = [(1, 6), (7, 12), (13, 999)]

def read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine="python")


def sanitize_filename(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-]+", "", text)
    return text or "untitled"


def ensure_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def filter_outliers_iqr(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    q1 = df.groupby(group_cols)[value_col].transform(lambda s: s.quantile(0.25))
    q3 = df.groupby(group_cols)[value_col].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    low = q1 - iqr_multiplier * iqr
    high = q3 + iqr_multiplier * iqr
    mask = df[value_col].between(low, high, inclusive="both")
    return df[mask].copy()


def add_group_zscore(
    df: pd.DataFrame,
    score_col: str = "target",
    group_cols: list[str] | tuple[str, ...] = ("src", "prompt"),
) -> pd.DataFrame:
    df = df.copy()
    for col in group_cols:
        if col not in df.columns:
            mean = df[score_col].mean()
            std = df[score_col].std(ddof=0)
            if std and std > 1e-12:
                df["target_norm"] = (df[score_col] - mean) / std
            else:
                df["target_norm"] = 0.0
            return df
    stats = (
        df.groupby(list(group_cols))[score_col]
        .agg(mean="mean", std=lambda s: float(s.std(ddof=0)))
        .reset_index()
    )
    df = df.merge(stats, on=list(group_cols), how="left")
    std = df["std"].replace(0.0, np.nan)
    df["target_norm"] = (df[score_col] - df["mean"]) / std
    df["target_norm"] = df["target_norm"].fillna(0.0)
    return df.drop(columns=["mean", "std"])  # cleanup


def plot_with_ci_shading(
    x: pd.Series,
    y_mean: pd.Series,
    y_sd: pd.Series,
    n: pd.Series,
    color: str,
    label: str,
) -> None:
    # 95% CI: mean ± 1.96 * SE, where SE = sd / sqrt(n)
    n_safe = np.maximum(n.astype(float).to_numpy(), 1.0)
    sd_vals = y_sd.astype(float).fillna(0.0).to_numpy()
    mean_vals = y_mean.astype(float).to_numpy()
    se = sd_vals / np.sqrt(n_safe)
    ci95 = 1.96 * se
    x_vals = x.astype(float).to_numpy()
    plt.plot(x_vals, mean_vals, marker="o", linewidth=2.0, alpha=0.95, color=color, label=label)
    plt.fill_between(x_vals, mean_vals - ci95, mean_vals + ci95, color=color, alpha=0.18, linewidth=0)


def plot_per_prompt(df: pd.DataFrame, out_dir: Path, title_suffix: str = "") -> int:
    prompts = sorted(p for p in df["prompt"].dropna().unique())
    count = 0
    for prompt in prompts:
        sub = df[df["prompt"] == prompt].copy()
        if sub.empty:
            continue
        sub["response_num"] = ensure_numeric(sub["response_num"])  # ensure numeric
        sub = sub.dropna(subset=["participant", "response_num", "target_norm"]).copy()
        if sub.empty:
            continue

        sub = sub[(sub["response_num"] >= 0) & (sub["response_num"] <= 18)].copy()
        sub = filter_outliers_iqr(sub, ["response_num"], "target_norm")
        if sub.empty:
            continue

        counts = sub.groupby("participant").size()
        total_participants = counts.index.nunique()
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(BINS)))
        plt.figure(figsize=(9, 6))
        for color, (lo, hi) in zip(colors, BINS):
            elig = set(counts[(counts >= lo) & (counts <= hi)].index)
            if not elig:
                continue
            df_bin = sub[sub["participant"].isin(elig)].copy()
            grp = (
                df_bin.groupby("response_num", as_index=False)
                      .agg(mean=("target_norm", "mean"), sd=("target_norm", "std"), n=("target_norm", "size"))
                      .sort_values("response_num")
            )
            if grp.empty:
                continue
            hi_cap = 18 if hi >= 999 else hi
            grp = grp[(grp["response_num"] >= 0) & (grp["response_num"] <= hi_cap)]
            if grp.empty:
                continue
            # Protect against NaN std for singletons
            grp["sd"] = grp["sd"].fillna(0.0)
            label_hi = "+" if hi >= 999 else f"–{hi}"
            label = f"{lo}{label_hi} responses (n={len(elig)})"
            plot_with_ci_shading(grp["response_num"], grp["mean"], grp["sd"], grp["n"], color=color, label=label)

        plt.xlabel("Response number (serial position)")
        plt.ylabel("Originality (z-score; within study and prompt)")
        t_suffix = f" — {title_suffix}" if title_suffix else ""
        plt.title(f"Stratified originality by response number (95% CI){t_suffix}\nPrompt: {prompt} (n={total_participants})", fontsize=11, pad=20)
        plt.xticks(list(range(0, 19)))
        plt.xlim(0, 18)
        plt.legend(frameon=False, fontsize=8, ncol=1, title="Cohorts by total responses per participant", title_fontsize=8)
        fname = out_dir / f"AUT_bins_prompt_{sanitize_filename(str(prompt))}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        count += 1
    return count


def plot_overall(df: pd.DataFrame, out_dir: Path, title_suffix: str = "") -> Path:
    df_clean = df[(df["response_num"] >= 0) & (df["response_num"] <= 18)].copy()
    df_clean = filter_outliers_iqr(df_clean, ["response_num"], "target_norm")
    if df_clean.empty:
        raise SystemExit("No data after outlier filtering for overall bins plot")

    counts = df_clean.groupby(["prompt", "participant"]).size()
    total_participants = df_clean["participant"].nunique()
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(BINS)))
    plt.figure(figsize=(9, 6))
    for color, (lo, hi) in zip(colors, BINS):
        elig_pairs = set(counts[(counts >= lo) & (counts <= hi)].index)
        if not elig_pairs:
            continue
        pairs_mask = df_clean.set_index(["prompt", "participant"]).index.isin(elig_pairs)
        df_bin = df_clean[pairs_mask].copy()
        grp = (
            df_bin.groupby("response_num", as_index=False)
                  .agg(mean=("target_norm", "mean"), sd=("target_norm", "std"), n=("target_norm", "size"))
                  .sort_values("response_num")
        )
        if grp.empty:
            continue
        hi_cap = 18 if hi >= 999 else hi
        grp = grp[(grp["response_num"] >= 0) & (grp["response_num"] <= hi_cap)]
        if grp.empty:
            continue
        grp["sd"] = grp["sd"].fillna(0.0)
        elig_participants = {p for (_, p) in elig_pairs}
        label_hi = "+" if hi >= 999 else f"–{hi}"
        label = f"{lo}{label_hi} responses (n={len(elig_participants)})"
        plot_with_ci_shading(grp["response_num"], grp["mean"], grp["sd"], grp["n"], color=color, label=label)

    plt.xlabel("Response number (serial position)")
    plt.ylabel("Originality (z-score; within study and prompt)")
    t_suffix = f" — {title_suffix}" if title_suffix else ""
    plt.title(f"Stratified originality by response number (95% CI){t_suffix}\nOverall participants (n={total_participants})", fontsize=11, pad=20)
    plt.xticks(list(range(0, 19)))
    plt.xlim(0, 18)
    plt.legend(frameon=False, fontsize=9, ncol=1, title="Cohorts by total responses per participant", title_fontsize=9)
    fname = out_dir / "AUT_bins_global.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def main():
    base_dir = Path(__file__).resolve().parent
    input_csv = base_dir / "Merged_AUT_Human_AI.csv"
    out_dir = base_dir / "figures_uses_bins_CI"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv_robust(input_csv)
    required = {"type", "prompt", "participant", "response_num", "target"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in {input_csv.name}: {sorted(missing)}")

    # Filter to uses only
    df = df[df["type"].astype(str).str.lower() == "uses"].copy()
    # Coerce numerics and drop nulls
    df["response_num"] = ensure_numeric(df["response_num"])  # may be float-like
    df["target"] = ensure_numeric(df["target"])              # human rating score
    df = df.dropna(subset=["prompt", "participant", "response_num", "target"]).copy()

    # Normalize per study+prompt
    df = add_group_zscore(df, score_col="target", group_cols=["src", "prompt"])
    # By-study outputs
    by_study_summary = {}
    if "src" in df.columns:
        # All studies combined
        all_dir = out_dir / "by_study" / "all"
        all_dir.mkdir(parents=True, exist_ok=True)
        c_all = plot_per_prompt(df, all_dir, title_suffix="All studies (uses; z-scored)")
        p_all = plot_overall(df, all_dir, title_suffix="All studies (uses; z-scored)")
        by_study_summary["all"] = {
            "per_prompt": c_all,
            "overall": str(p_all),
        }

        for study in sorted(df["src"].dropna().unique()):
            sub = df[df["src"] == study].copy()
            study_dir = out_dir / "by_study" / sanitize_filename(str(study))
            study_dir.mkdir(parents=True, exist_ok=True)
            c_s = plot_per_prompt(sub, study_dir, title_suffix=f"Study: {study} (uses; z-scored)")
            p_s = plot_overall(sub, study_dir, title_suffix=f"Study: {study} (uses; z-scored)")
            by_study_summary[str(study)] = {
                "per_prompt": c_s,
                "overall": str(p_s),
            }

    print({
        "input": str(input_csv),
        "figures_dir": str(out_dir),
        "by_study": by_study_summary,
    })


if __name__ == "__main__":
    main()