# Usage Guide

## Overview

This guide explains how to use the IoT Device Fingerprinting with NYM Mixnet repository for both baseline IoT analysis and NYM mixnet privacy assessment.

## Repository Structure

```
IoT_NYM_Thesis_Repository/
├── iot_baseline_analysis/          # Baseline IoT device fingerprinting
│   ├── notebooks/                  # Jupyter notebooks
│   │   └── 1.IoT_Fingerprint_ML_Evaluation.ipynb
│   ├── data/                       # IoT baseline data
│   │   ├── raw/                    # PCAP files
│   │   └── processed/              # CSV files
│   └── figures/                    # Generated plots (14 files)
├── nym_mixnet_analysis/            # NYM mixnet privacy analysis
│   ├── scripts/                    # Core pipeline scripts
│   │   ├── nym_pipeline.py
│   │   └── run_nym_unsup.py
│   ├── nym_unsup/                  # Analysis directory
│   │   ├── data/                   # NYM data files
│   │   │   ├── raw/                # PCAP files
│   │   │   └── processed/          # CSV files
│   │   ├── requirements.txt        # Dependencies
│   │   └── README.md               # Analysis documentation
│   ├── results/                    # Analysis results
│   │   ├── nym_mixnet_analysis_results.md
│   │   ├── comparison_summary.md
│   │   └── tables/                 # CSV result files
│   └── README.md                   # NYM analysis documentation
└── documentation/                  # This guide
```

## IoT Baseline Analysis

### Purpose
Demonstrates traditional IoT device fingerprinting using flow-level metadata features without privacy protection.

### Key Features
- **14 Flow-level Features**: Duration, packet counts, timing statistics
- **Multiple ML Models**: Random Forest, SVM, Decision Tree, KNN
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score
- **Feature Importance Analysis**: Understanding discriminative features
- **Rich Visualizations**: 14 generated plots for analysis

### Running the Analysis

1. **Navigate to the IoT analysis directory**:
   ```bash
   cd iot_baseline_analysis/notebooks
   ```

2. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

3. **Open the analysis notebook**:
   - Open `1.IoT_Fingerprint_ML_Evaluation.ipynb`
   - Run all cells sequentially

4. **Expected Results**:
   - Model performance comparison table
   - Confusion matrices
   - Feature importance plots
   - Classification reports
   - 14 generated visualization files

### Output Files
- `iot_baseline_analysis/figures/` - 14 generated plots
- Model performance results in notebook
- Classification reports and metrics

## NYM Mixnet Analysis

### Purpose
Evaluates privacy preservation effectiveness of NYM mixnet technology on IoT device fingerprinting.

### Key Features
- **Unsupervised Clustering**: HDBSCAN, GMM-BIC, Spectral Clustering
- **Privacy Assessment**: Noise detection, cluster stability
- **Proxy Classification**: Pseudo-label classification for separability
- **Comparative Analysis**: Baseline vs. mixnet performance

### Running the Analysis

1. **Navigate to the NYM analysis directory**:
   ```bash
   cd nym_mixnet_analysis/scripts
   ```

2. **Run the analysis pipeline**:
   ```bash
   python run_nym_unsup.py
   ```

3. **Expected Output**:
   ```
   Loading dataset...
   Running HDBSCAN/DBSCAN clustering...
   Computing stability...
   [HDBSCAN/DBSCAN] clusters: 10, noise: 0.256
   [HDBSCAN/DBSCAN] Silhouette: 0.906, DB: 0.199
   ...
   Analysis completed successfully!
   ```

### Output Files
- `nym_mixnet_analysis/results/tables/clustering_summary.csv` - Clustering performance metrics
- `nym_mixnet_analysis/results/tables/proxy_classification_results.csv` - Classification results
- `nym_mixnet_analysis/results/tables/confusion_matrix.csv` - Detailed classification matrix
- `nym_mixnet_analysis/results/nym_mixnet_analysis_results.md` - Clean results summary

## Data Requirements

### Raw Data (PCAP Files)
- **IoT Baseline**: Available in `iot_baseline_analysis/data/raw/` - Original IoT traffic captures
- **NYM Mixnet**: Available in `nym_mixnet_analysis/nym_unsup/data/raw/` - NYM mixnet traffic captures

### Processed Data (CSV Files)
- **IoT Baseline**: Available in `iot_baseline_analysis/data/processed/` - Extracted flow-level features
- **NYM Mixnet**: Available in `nym_mixnet_analysis/nym_unsup/data/processed/` - Extracted flow-level features

### Data Processing Pipeline
1. **PCAP Capture**: Network traffic captured using Wireshark/tcpdump
2. **Flow Extraction**: CICFlowMeter used to extract flow-level metadata
3. **Feature Engineering**: 14 specific features selected for analysis
4. **Data Cleaning**: Missing values and outliers handled appropriately

## Results Interpretation

### Baseline Analysis Results
- **High Accuracy (>95%)**: Indicates effective device fingerprinting
- **Feature Importance**: Shows which features are most discriminative
- **Model Comparison**: Identifies best performing algorithms
- **Visualizations**: 14 plots for comprehensive analysis

### NYM Mixnet Results
- **Cluster Count**: Number of discovered device patterns
- **Noise Rate**: Percentage of unclassified flows (privacy indicator)
- **Silhouette Score**: Cluster quality (higher = better separation)
- **Proxy Classification**: Separability of artificial patterns

## Customization

### Modifying Feature Set
Edit the `features` list in the analysis scripts:
```python
features = [
    "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    # Add or remove features as needed
]
```

### Adjusting Model Parameters
Modify model configurations in the scripts:
```python
# Example: Change Random Forest parameters
RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
```

### Adding New Models
Add new models to the analysis:
```python
from sklearn.ensemble import GradientBoostingClassifier

models["Gradient Boosting"] = GradientBoostingClassifier(random_state=42)
```

## Troubleshooting

### Common Issues

1. **Data Loading Errors**:
   - Check file paths in `data/processed/`
   - Verify CSV format and column names
   - Ensure no missing values in required features

2. **Memory Issues**:
   - Reduce dataset size for testing
   - Use smaller model parameters
   - Close other applications

3. **Import Errors**:
   - Verify all dependencies are installed
   - Check Python version compatibility
   - Reinstall problematic packages

### Getting Help

1. Check the installation guide in `documentation/setup/`
2. Review error messages for specific issues
3. Verify data format matches requirements
4. Test with smaller datasets first

## Advanced Usage

### Batch Processing
Run multiple analyses:
```bash
# Run both analyses
cd iot_baseline_analysis/notebooks && jupyter nbconvert --to script 1.IoT_Fingerprint_ML_Evaluation.ipynb
cd ../../nym_mixnet_analysis/scripts && python run_nym_unsup.py
```

### Custom Evaluation
Add custom evaluation metrics:
```python
from sklearn.metrics import roc_auc_score, average_precision_score

# Add to evaluation pipeline
auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
ap_score = average_precision_score(y_test, y_pred_proba)
```

### Exporting Results
Save results in different formats:
```python
# Save as Excel
results_df.to_excel('results.xlsx', index=False)

# Save as JSON
results_dict.to_json('results.json', indent=2)
```
