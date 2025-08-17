# NYM Mixnet Privacy Analysis

## Overview

This directory contains the NYM mixnet privacy preservation analysis using Python scripts (not Jupyter notebooks). The analysis evaluates how NYM mixnet technology obfuscates IoT device fingerprints through unsupervised clustering.

## Key Features

- **Unsupervised Clustering**: HDBSCAN, GMM-BIC, Spectral Clustering
- **Privacy Assessment**: Noise detection, cluster stability analysis
- **Proxy Classification**: Pseudo-label classification for separability
- **Python Scripts**: Direct execution (no Jupyter notebooks)

## File Structure

```
nym_mixnet_analysis/
├── scripts/
│   ├── nym_pipeline.py          # Core clustering algorithms
│   └── run_nym_unsup.py         # Main execution script
├── notebooks/                   # (Empty - uses Python scripts)
└── README.md                    # This file
```

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Analysis
```bash
cd scripts
python run_nym_unsup.py
```

### Expected Output
```
Loading dataset...
Running HDBSCAN/DBSCAN clustering...
Computing stability...
[HDBSCAN/DBSCAN] clusters: 10, noise: 0.256
[HDBSCAN/DBSCAN] Silhouette: 0.906, DB: 0.199
...
Analysis completed successfully!
```

## Output Files

The analysis generates:
- `clustering_summary.csv` - Clustering performance metrics
- `proxy_classification_results.csv` - Classification results
- `confusion_matrix.csv` - Detailed classification matrix

## Key Results

- **HDBSCAN**: 10 clusters, 25.58% noise rate
- **Silhouette Score**: 0.906 (excellent separation)
- **Davies-Bouldin**: 0.199 (excellent cluster quality)
- **Stability**: Bootstrap ARI 0.775

## Data Requirements

- **File**: `data/raw/nym_metadata.csv`
- **Format**: CSV with 14 flow-level features
- **Source**: NYM mixnet traffic from Raspberry Pi 5 entry gateway
- **Samples**: 1,642 network flows

## Algorithm Details

### HDBSCAN Clustering
- Density-based clustering with noise detection
- Automatic parameter selection
- Handles varying density clusters

### GMM-BIC
- Gaussian Mixture Model with Bayesian Information Criterion
- Automatic model selection
- Probabilistic clustering

### Spectral Clustering
- Graph-based clustering
- Eigengap heuristic for k selection
- Non-linear cluster boundaries

## Privacy Assessment

The analysis demonstrates:
- **Device Identity Obfuscation**: 2 original devices → 10 apparent clusters
- **Traffic Pattern Transformation**: Artificial pattern creation
- **Noise Introduction**: 25.58% noise rate enhances privacy
- **Stability**: Consistent privacy protection across samples

## Comparison with Baseline

This analysis should be compared with the IoT baseline analysis to demonstrate:
- Privacy preservation effectiveness
- Device fingerprinting vulnerability reduction
- Trade-offs between privacy and detection capabilities
