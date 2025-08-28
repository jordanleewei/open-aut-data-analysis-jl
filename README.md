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

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure your data file `Merged_AUT_Human_AI.csv` is in the same directory as the script
2. Run the analysis:

```bash
python AUT_stratified_bins_with_95_CI.py
```

## Output

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

## Statistical Methods

- **Normalization**: Z-scores calculated within study and prompt groups
- **Outlier Detection**: IQR method with 1.5x multiplier
- **Confidence Intervals**: 95% CI using standard error (SE = SD/√n)
- **Grouping**: Participants stratified by total response counts into three bins

## Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation and analysis
- `matplotlib`: Plotting and visualization

## File Structure

```
├── AUT_stratified_bins_with_95_CI.py    # Main analysis script
├── Merged_AUT_Human_AI.csv              # Input data file
├── requirements.txt                      # Python dependencies
├── README.md                            # This file
└── figures_uses_bins_CI/                # Generated output directory
    ├── AUT_bins_global.png              # Overall analysis
    ├── by_study/                        # Study-specific analyses
    │   ├── all/                         # All studies combined
    │   └── [study_name]/                # Individual study results
    └── AUT_bins_prompt_[prompt].png     # Per-prompt analyses
```
