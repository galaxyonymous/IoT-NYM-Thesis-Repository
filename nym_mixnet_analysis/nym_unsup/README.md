# NYM Mixnet Unsupervised Analysis

This directory contains the NYM mixnet unsupervised clustering analysis for IoT device privacy preservation assessment.

## Overview

The analysis evaluates the effectiveness of NYM mixnet technology in obfuscating IoT device fingerprints through unsupervised clustering techniques. It compares multiple clustering algorithms and assesses privacy preservation through proxy classification.

## File Structure

```
nym_unsup/
├── scripts/                    # Core analysis scripts (in parent directory)
│   ├── nym_pipeline.py         # All reusable functions
│   └── run_nym_unsup.py        # Main execution script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/                       # Data directory
│   ├── raw/                    # Raw data files
│   │   └── NYM_Mixnet_IoT_matadata.pcap
│   └── processed/              # Processed data files
│       ├── nym_mixnet_IoT_metadata_subset.csv
│       └── nym_metadata.csv
└── __pycache__/                # Python cache files
```

## Key Components

### Core Scripts
- **`../scripts/nym_pipeline.py`**: Contains all clustering algorithms (HDBSCAN, GMM-BIC, Spectral) and evaluation functions
- **`../scripts/run_nym_unsup.py`**: Main execution script that orchestrates the complete analysis pipeline

### Data Files
- **`data/processed/nym_mixnet_IoT_metadata_subset.csv`**: Main NYM mixnet dataset (1,642 flows, 14 features)
- **`data/processed/nym_metadata.csv`**: Alternative NYM dataset format
- **`data/raw/NYM_Mixnet_IoT_matadata.pcap`**: Original NYM mixnet traffic capture

### Output Files
- **`../results/tables/clustering_summary.csv`**: Performance metrics for all clustering methods
- **`../results/tables/proxy_classification_results.csv`**: Classification performance on pseudo-labels
- **`../results/tables/confusion_matrix.csv`**: Detailed confusion matrix for device distribution
- **`../results/nym_mixnet_analysis_results.md`**: Clean results summary

## Usage

### Running the Analysis

1. **Navigate to the scripts directory**:
   ```bash
   cd ../scripts
   ```

2. **Run the main analysis**:
   ```bash
   python run_nym_unsup.py
   ```

3. **Expected Output**:
   ```
   Loading dataset...
   Running HDBSCAN/DBSCAN clustering...
   [HDBSCAN/DBSCAN] clusters: 10, noise: 0.256
   [HDBSCAN/DBSCAN] Silhouette: 0.906, DB: 0.199
   ...
   Analysis completed successfully!
   ```

### Key Results

- **Best Method**: HDBSCAN (10 clusters, 25.58% noise rate)
- **Cluster Quality**: Excellent separation (Silhouette = 0.906)
- **Stability**: Good bootstrap stability (ARI = 0.767)
- **Proxy Classification**: 99.73% accuracy (Random Forest)

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- pandas
- scikit-learn
- matplotlib
- hdbscan

## Analysis Pipeline

1. **Data Loading**: Load and preprocess NYM mixnet metadata from `data/processed/`
2. **Feature Standardization**: Normalize features to mean=0, std=1
3. **Clustering Analysis**: Run HDBSCAN, GMM-BIC, and Spectral clustering
4. **Stability Assessment**: Compute bootstrap ARI for cluster stability
5. **Proxy Classification**: Train ML models on pseudo-labels
6. **Results Generation**: Save performance metrics and confusion matrices

## Privacy Assessment

The analysis demonstrates:
- **Device Identity Obfuscation**: 2 original devices → 10 apparent devices
- **Noise Enhancement**: 25.58% noise rate increases privacy
- **Pattern Separation**: Artificial traffic patterns mask real device characteristics

## Troubleshooting

- **HDBSCAN Installation**: `pip install hdbscan` (auto-falls back to DBSCAN if not available)
- **Missing Columns**: Check header normalization in data files
- **Memory Issues**: Reduce dataset size for testing
- **File Paths**: Ensure data files are in `data/processed/` directory

## Repository Integration

This analysis is part of the larger thesis repository:
- **Main Repository**: `IoT_NYM_Thesis_Repository/`
- **Scripts Location**: `nym_mixnet_analysis/scripts/`
- **Results**: Available in `nym_mixnet_analysis/results/nym_mixnet_analysis_results.md`
- **Documentation**: See `documentation/usage/USAGE_GUIDE.md`
- **Baseline Analysis**: See `iot_baseline_analysis/notebooks/1.IoT_Fingerprint_ML_Evaluation.ipynb`

## Data Sources

- **NYM Mixnet Data**: `data/processed/nym_mixnet_IoT_metadata_subset.csv` (1,642 flows)
- **Alternative Format**: `data/processed/nym_metadata.csv` (same data, different format)
- **Raw Data**: Available in `data/raw/NYM_Mixnet_IoT_matadata.pcap` (PCAP file)

---

**Note**: This analysis focuses on unsupervised clustering for privacy assessment. For baseline IoT device fingerprinting, see the `iot_baseline_analysis/` directory.

**Author**: Muhammad Siddique (muhammad.siddique@ulb.be)  
**Repository**: https://github.com/galaxyonymous/IoT-NYM-Thesis-Repository
