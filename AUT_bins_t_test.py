from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


# Bins : [1–6], [>7] (combining 7-12 and 13+)
BINS: list[tuple[int, int]] = [(1, 6), (7, 999)]


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read CSV file with robust error handling."""
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine="python")


def sanitize_filename(text: str) -> str:
    """Sanitize text for use in filenames."""
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-]+", "", text)
    return text or "untitled"


def ensure_numeric(series: pd.Series) -> pd.Series:
    """Ensure series is numeric, coercing errors to NaN."""
    return pd.to_numeric(series, errors="coerce")


def filter_outliers_iqr(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Filter outliers using IQR method."""
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
    """Add z-scored target column normalized within groups."""
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
    return df.drop(columns=["mean", "std"])


def assign_bin(participant_counts: pd.Series, participant_key) -> str:
    """Assign a participant to a bin based on their response count."""
    count = participant_counts.get(participant_key, 0)
    for lo, hi in BINS:
        if lo <= count <= hi:
            if hi >= 999:
                return ">7"
            else:
                return f"{lo}-{hi}"
    return "unknown"


def get_bin_hi(bin_name: str) -> int:
    """Return the maximum response position to include for a bin, mirroring viz logic."""
    if bin_name == "1-6":
        return 6
    return 999


def perform_t_tests(bin_data: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Perform pairwise t-tests between all bins."""
    bin_names = list(bin_data.keys())
    results = []
    
    for i, bin1 in enumerate(bin_names):
        for j, bin2 in enumerate(bin_names):
            if i >= j:  # Skip self-comparisons and avoid duplicates
                continue
                
            data1 = bin_data[bin1]
            data2 = bin_data[bin2]
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                 (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                (len(data1) + len(data2) - 2))
            if pooled_std > 0:
                cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
            else:
                cohens_d = 0.0
            
            results.append({
                'bin1': bin1,
                'bin2': bin2,
                'n1': len(data1),
                'n2': len(data2),
                'mean1': np.mean(data1),
                'mean2': np.mean(data2),
                'std1': np.std(data1, ddof=1),
                'std2': np.std(data2, ddof=1),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05,
                'highly_significant': p_value < 0.01
            })
    
    return pd.DataFrame(results)


def analyze_bins_by_response_position(df: pd.DataFrame) -> Dict:
    """Analyze bins with respect to response position (serial order)."""
    # Clean and prepare data
    df_clean = df[(df["response_num"] >= 0) & (df["response_num"] <= 18)].copy()
    df_clean = filter_outliers_iqr(df_clean, ["response_num"], "target_norm")
    
    if df_clean.empty:
        print("No data after outlier filtering")
        return {}
    
    # Get participant response counts per prompt (matching original script)
    counts = df_clean.groupby(["prompt", "participant"]).size()
    
    # Assign bins to participants
    df_clean['bin'] = df_clean.apply(lambda row: assign_bin(counts, (row['prompt'], row['participant'])), axis=1)
    
    # Remove unknown bins
    df_clean = df_clean[df_clean['bin'] != "unknown"].copy()
    
    if df_clean.empty:
        print("No valid bin assignments found")
        return {}
    
    # Analyze by response position
    response_positions = sorted(df_clean['response_num'].unique())
    position_results = []
    
    print(f"\n{'='*80}")
    print(f"BIN ANALYSIS BY RESPONSE POSITION")
    print(f"{'='*80}")
    
    for pos in response_positions:
        pos_data = df_clean[df_clean['response_num'] == pos]
        
        if pos_data.empty:
            continue
            
        # Group by bins for this position, enforcing per-bin max response position
        bin_data = {}
        for bin_name in pos_data['bin'].unique():
            # Enforce upper cap like in visualization (e.g., 1-6 only up to position 6)
            if pos > get_bin_hi(bin_name):
                continue
            bin_values = pos_data[pos_data['bin'] == bin_name]['target_norm'].values
            if len(bin_values) >= 3:  # Require at least 3 observations per bin
                bin_data[bin_name] = bin_values
        
        if len(bin_data) < 2:
            continue
            
        # Perform t-tests for this position
        t_test_results = perform_t_tests(bin_data)
        
        # Add position information
        t_test_results['response_position'] = pos
        
        # Calculate summary statistics for this position
        summary_stats = {}
        for bin_name, data in bin_data.items():
            summary_stats[bin_name] = {
                'n': len(data),
                'mean': np.mean(data),
                'std': np.std(data, ddof=1),
                'se': np.std(data, ddof=1) / np.sqrt(len(data))
            }
        
        # Print results for this position
        print(f"\nResponse Position {pos}:")
        print("-" * 50)
        
        # Print summary statistics
        for bin_name, stats in summary_stats.items():
            print(f"  {bin_name:>8}: n={stats['n']:3d}, mean={stats['mean']:6.3f}, "
                  f"std={stats['std']:6.3f}, se={stats['se']:6.3f}")
        
        # Print t-test results
        if not t_test_results.empty:
            print("  T-tests:")
            for _, row in t_test_results.iterrows():
                sig = "***" if row['highly_significant'] else "**" if row['significant'] else ""
                print(f"    {row['bin1']} vs {row['bin2']:<8}: t={row['t_statistic']:6.3f}, "
                      f"p={row['p_value']:8.4f}, d={row['cohens_d']:6.3f} {sig}")
        
        position_results.append({
            'position': pos,
            't_test_results': t_test_results,
            'summary_stats': summary_stats
        })
    
    return {
        'position_results': position_results,
        'total_positions': len(position_results)
    }


def create_visualization(results: Dict, output_dir: Path) -> None:
    """Create visualization plots for the t-test results."""
    if not results or 'position_results' not in results:
        return
    
    # Extract data for plotting
    positions = []
    means_1_6 = []
    means_7_plus = []
    ses_1_6 = []
    ses_7_plus = []
    p_values = []
    
    for pos_result in results['position_results']:
        pos = pos_result['position']
        summaries = pos_result['summary_stats']
        
        if '1-6' in summaries and '>7' in summaries:
            positions.append(pos)
            means_1_6.append(summaries['1-6']['mean'])
            means_7_plus.append(summaries['>7']['mean'])
            ses_1_6.append(summaries['1-6']['se'])
            ses_7_plus.append(summaries['>7']['se'])
            
            # Get p-value from t-test results
            t_tests = pos_result['t_test_results']
            if not t_tests.empty:
                p_values.append(t_tests.iloc[0]['p_value'])
            else:
                p_values.append(1.0)
    
    if not positions:
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Convert to numpy arrays
    positions = np.array(positions)
    means_1_6 = np.array(means_1_6)
    means_7_plus = np.array(means_7_plus)
    ses_1_6 = np.array(ses_1_6)
    ses_7_plus = np.array(ses_7_plus)
    p_values = np.array(p_values)
    
    # Plot lines with error bars
    plt.errorbar(positions, means_1_6, yerr=ses_1_6, 
                marker='o', linewidth=2, markersize=8, 
                label='1-6 responses', color='#E74C3C', capsize=5)
    plt.errorbar(positions, means_7_plus, yerr=ses_7_plus, 
                marker='s', linewidth=2, markersize=8, 
                label='>7 responses', color='#3498DB', capsize=5)
    
    # Add significance indicators
    for i, (pos, p_val) in enumerate(zip(positions, p_values)):
        if p_val < 0.001:
            plt.text(pos, max(means_1_6[i], means_7_plus[i]) + 0.05, '***', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        elif p_val < 0.01:
            plt.text(pos, max(means_1_6[i], means_7_plus[i]) + 0.05, '**', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        elif p_val < 0.05:
            plt.text(pos, max(means_1_6[i], means_7_plus[i]) + 0.05, '*', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Customize the plot
    plt.xlabel('Response Position (Serial Order)', fontsize=14)
    plt.ylabel('Originality (z-score; within study and prompt)', fontsize=14)
    plt.title('Originality by Response Position: 1-6 vs >7 Responses\n(Error bars = Standard Error, *p<0.05, **p<0.01, ***p<0.001)', 
              fontsize=16, pad=20)
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.xticks(positions, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Adjust layout and save
    plt.tight_layout()
    plot_file = output_dir / "originality_comparison_1_6_vs_7_plus.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {plot_file}")
    
    # Create a summary table visualization
    create_summary_table(results, output_dir)


def create_summary_table(results: Dict, output_dir: Path) -> None:
    """Create a summary table visualization showing key statistics."""
    if not results or 'position_results' not in results:
        return
    
    # Extract data for table
    table_data = []
    for pos_result in results['position_results']:
        pos = pos_result['position']
        summaries = pos_result['summary_stats']
        t_tests = pos_result['t_test_results']
        
        if '1-6' in summaries and '>7' in summaries:
            row = {
                'Position': pos,
                '1-6_n': summaries['1-6']['n'],
                '1-6_mean': summaries['1-6']['mean'],
                '1-6_se': summaries['1-6']['se'],
                '>7_n': summaries['>7']['n'],
                '>7_mean': summaries['>7']['mean'],
                '>7_se': summaries['>7']['se'],
                't_stat': t_tests.iloc[0]['t_statistic'] if not t_tests.empty else np.nan,
                'p_value': t_tests.iloc[0]['p_value'] if not t_tests.empty else np.nan,
                'cohens_d': t_tests.iloc[0]['cohens_d'] if not t_tests.empty else np.nan
            }
            table_data.append(row)
    
    if not table_data:
        return
    
    # Create DataFrame and format for display
    df = pd.DataFrame(table_data)
    
    # Format the data for better display
    df['1-6_mean_se'] = df['1-6_mean'].round(3).astype(str) + ' ± ' + df['1-6_se'].round(3).astype(str)
    df['>7_mean_se'] = df['>7_mean'].round(3).astype(str) + ' ± ' + df['>7_se'].round(3).astype(str)
    df['t_stat'] = df['t_stat'].round(3)
    df['p_value'] = df['p_value'].round(4)
    df['cohens_d'] = df['cohens_d'].round(3)
    
    # Add significance indicators
    df['significance'] = ''
    df.loc[df['p_value'] < 0.001, 'significance'] = '***'
    df.loc[(df['p_value'] >= 0.001) & (df['p_value'] < 0.01), 'significance'] = '**'
    df.loc[(df['p_value'] >= 0.01) & (df['p_value'] < 0.05), 'significance'] = '*'
    
    # Select columns for display
    display_df = df[['Position', '1-6_n', '1-6_mean_se', '>7_n', '>7_mean_se', 
                     't_stat', 'p_value', 'cohens_d', 'significance']].copy()
    display_df.columns = ['Pos', '1-6_n', '1-6_mean±se', '>7_n', '>7_mean±se', 
                          't-stat', 'p-value', "Cohen's d", 'Sig']
    
    # Save formatted table
    table_file = output_dir / "summary_table_formatted.csv"
    display_df.to_csv(table_file, index=False)
    
    print(f"Summary table saved: {table_file}")
    
    # Print formatted table to console
    print(f"\n{'='*100}")
    print("SUMMARY TABLE: 1-6 vs >7 Responses by Position")
    print(f"{'='*100}")
    print(display_df.to_string(index=False, float_format='%.3f'))
    print(f"{'='*100}")
    print("Note: *** p<0.001, ** p<0.01, * p<0.05")
    print(f"{'='*100}")


def save_position_results(results: Dict, output_dir: Path) -> None:
    """Save response position analysis results to CSV files."""
    if not results or 'position_results' not in results:
        return
    
    # Combine all t-test results
    all_t_tests = []
    all_summaries = []
    
    for pos_result in results['position_results']:
        pos = pos_result['position']
        t_tests = pos_result['t_test_results']
        summaries = pos_result['summary_stats']
        
        # Add position to t-test results
        if not t_tests.empty:
            all_t_tests.append(t_tests)
        
        # Convert summary stats to DataFrame
        summary_df = pd.DataFrame(summaries).T
        summary_df.index.name = 'bin'
        summary_df.reset_index(inplace=True)
        summary_df['response_position'] = pos
        all_summaries.append(summary_df)
    
    # Save combined results
    if all_t_tests:
        combined_t_tests = pd.concat(all_t_tests, ignore_index=True)
        t_test_file = output_dir / "t_test_results_by_position.csv"
        combined_t_tests.to_csv(t_test_file, index=False)
        print(f"\nT-test results saved: {t_test_file}")
    
    if all_summaries:
        combined_summaries = pd.concat(all_summaries, ignore_index=True)
        summary_file = output_dir / "summary_stats_by_position.csv"
        combined_summaries.to_csv(summary_file, index=False)
        print(f"Summary statistics saved: {summary_file}")


def main():
    """Main function to run t-test analysis by response position."""
    base_dir = Path(__file__).resolve().parent
    input_csv = base_dir / "Merged_AUT_Human_Rating.csv"
    out_dir = base_dir / "t_test_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("AUT Bins T-Test Analysis by Response Position")
    print("=" * 50)
    
    # Read and prepare data
    df = read_csv_robust(input_csv)
    required = {"type", "prompt", "participant", "response_num", "target"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in {input_csv.name}: {sorted(missing)}")
    
    # Filter to uses only
    df = df[df["type"].astype(str).str.lower() == "uses"].copy()
    df["response_num"] = ensure_numeric(df["response_num"])
    df["target"] = ensure_numeric(df["target"])
    df = df.dropna(subset=["prompt", "participant", "response_num", "target"]).copy()
    
    print(f"Total data points: {len(df)}")
    print(f"Unique participants: {df['participant'].nunique()}")
    print(f"Response positions: {sorted(df['response_num'].unique())}")
    
    # Normalize per study+prompt
    df = add_group_zscore(df, score_col="target", group_cols=["src", "prompt"])
    
    # Analyze bins by response position
    results = analyze_bins_by_response_position(df)
    
    # Save results
    save_position_results(results, out_dir)
    
    # Create visualization
    create_visualization(results, out_dir)
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"Results saved to: {out_dir}")
    print(f"Total response positions analyzed: {results.get('total_positions', 0)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
