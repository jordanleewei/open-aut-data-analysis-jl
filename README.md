# AUT Stratified Bins Analysis with 95% Confidence Intervals

This project analyzes Alternative Uses Task (AUT) data to examine originality patterns across different response positions, stratified by participant response counts.

## Overview

The analysis focuses on the "uses" responses from AUT studies, examining how originality scores vary across serial positions (response numbers 1-18) for different cohorts of participants based on their total response counts. The analysis creates visualizations with 95% confidence intervals to show statistical significance.

## Features

- **Stratified Analysis**: Groups participants into bins based on total response counts:
  - 1-6 responses
  - 7-12 responses
  - 13+ responses
- **Confidence Intervals**: Calculates and visualizes 95% confidence intervals for originality scores
- **Per-Prompt Analysis**: Generates separate plots for each AUT prompt
- **Study-Level Analysis**: Provides both overall and study-specific visualizations
- **Outlier Filtering**: Uses IQR-based outlier detection to ensure robust analysis
- **Z-Score Normalization**: Normalizes originality scores within study and prompt groups

## Data Requirements

The analysis expects a CSV file (`Merged_AUT_Human_AI.csv`) with the following required columns:

- `type`: Response type (filtered to "uses" only)
- `src`: Study source identifier
- `prompt`: AUT prompt/question
- `participant`: Participant identifier
- `response_num`: Serial position of the response (e.g. 0-18)
- `target`: Originality rating score

## Dataset Details

The file `Details of AUT Open Dataset used in OSCAI.xlsx` provides a brief reference to the AUT Open dataset used in this project. It includes:

- Original study
- Participant counts and response counts
- Participants and rater instructions
- Rater counts
- Links to Datasets and Papers

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Visualization Analysis

1. Ensure your data file `Merged_AUT_Human_AI.csv` is in the same directory as the script
2. Run the visualization analysis:

```bash
python AUT_stratified_bins_with_95_CI.py
```

### Statistical Testing

To perform t-tests between the response count bins:

```bash
python AUT_bins_t_test.py
```

## Output

### Visualization Analysis

The script generates a `figures_uses_bins_CI` directory containing:

- **Overall analysis**: `AUT_bins_global.png` - Combined analysis across all studies
- **Per-prompt analysis**: `AUT_bins_prompt_[prompt_name].png` - Individual plots for each AUT prompt
- **Study-specific analysis**: Separate directories for each study with their respective plots

Each plot shows:

- Originality scores (z-scored) on the y-axis
- Response number (serial position 0-18) on the x-axis
- Different colored lines for each response count cohort
- 95% confidence intervals as shaded areas
- Participant counts for each cohort

### Statistical Testing

The t-test script generates a `t_test_results` directory containing:

- **T-test results**: `t_test_results_[analysis_name].csv` - Pairwise t-tests between bins
- **Summary statistics**: `summary_stats_[analysis_name].csv` - Descriptive statistics for each bin

The analysis includes:

- Overall analysis across all studies
- Study-specific analyses
- Prompt-specific analyses (first 5 prompts)
- Effect sizes (Cohen's d)
- Significance indicators (p < 0.05, p < 0.01)

## Statistical Methods

- **Normalization**: Z-scores calculated within study and prompt groups
- **Outlier Detection**: IQR method with 1.5x multiplier
- **Confidence Intervals**: 95% CI using standard error (SE = SD/√n)
- **Grouping**: Participants stratified by total response counts into three bins

## Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation and analysis
- `matplotlib`: Plotting and visualization
- `scipy`: Statistical testing (t-tests, effect sizes)

## File Structure

```
├── AUT_stratified_bins_with_95_CI.py    # Main visualization script
├── AUT_bins_t_test.py                   # Statistical testing script
├── Merged_AUT_Human_AI.csv              # Input data file
├── requirements.txt                      # Python dependencies
├── README.md                            # This file
├── figures_uses_bins_CI/                # Visualization output directory
│   ├── AUT_bins_global.png              # Overall analysis
│   ├── by_study/                        # Study-specific analyses
│   │   ├── all/                         # All studies combined
│   │   └── [study_name]/                # Individual study results
│   └── AUT_bins_prompt_[prompt].png     # Per-prompt analyses
└── t_test_results/                      # Statistical testing output directory
    ├── t_test_results_[analysis].csv    # T-test results
    └── summary_stats_[analysis].csv     # Summary statistics
```
