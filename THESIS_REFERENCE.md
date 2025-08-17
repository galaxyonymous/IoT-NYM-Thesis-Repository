# Thesis Reference Guide

## Overview

This repository serves as the complete implementation reference for the thesis:

**Title**: IoT Devices Fingerprinting based on Metadata Classification, Threat Models and Countermeasures

## Chapter Mapping

### Chapter 4: IoT Device Fingerprinting Baseline Analysis

**Repository Location**: `iot_baseline_analysis/`

**Key Components**:
- **Notebook**: `iot_baseline_analysis/notebooks/1.IoT_Fingerprint_ML_Evaluation.ipynb`
- **Data**: Available in `iot_baseline_analysis/data/`
- **Results**: Available in `nym_mixnet_analysis/results/tables/baseline_classification_results.csv`

**Thesis Content**:
- Traditional IoT device fingerprinting using flow-level metadata
- Supervised learning with multiple ML algorithms
- Feature importance analysis
- Performance evaluation metrics

**Repository Files**:
```
iot_baseline_analysis/
├── notebooks/
│   └── 1.IoT_Fingerprint_ML_Evaluation.ipynb    # Complete baseline analysis
├── data/
│   ├── raw/                                      # PCAP files
│   │   ├── Sample_Metadata_Smart_Plug.pcap
│   │   ├── Sample_Metadata_lamp.pcap
│   │   └── IoT_Network_Devices_MetaData_RPI_GW.pcap
│   └── processed/                                # CSV files
│       ├── Sample_Metadata_Smart_Plug.csv
│       ├── Sample_Metadata_Smart_Lamp.csv
│       └── IoT_metadata.csv
└── figures/                                      # Generated plots (14 files)
    ├── baseline_scaling_effect_barh.png
    ├── baseline_learning_curves_single.png
    ├── baseline_pr_2.png
    ├── baseline_pr.png
    ├── baseline_roc.png
    ├── baseline_learning_curves_2x2.png
    ├── baseline_cluster_sizes.png
    ├── baseline_kmeans_silhouette.png
    ├── baseline_kmeans_elbow.png
    ├── baseline_perm_importance_svm.png
    ├── baseline_rf_importance.png
    ├── baseline_cv_boxplot.png
    ├── baseline_confusion_grid.png
    └── baseline-metrics-grouped.png
```

### Chapter 5: NYM Mixnet Privacy Preservation Assessment

**Repository Location**: `nym_mixnet_analysis/`

**Key Components**:
- **Pipeline**: `nym_mixnet_analysis/scripts/nym_pipeline.py`
- **Execution**: `nym_mixnet_analysis/scripts/run_nym_unsup.py`
- **Data**: `nym_mixnet_analysis/nym_unsup/data/processed/nym_mixnet_IoT_metadata_subset.csv`
- **Results**: `nym_mixnet_analysis/results/nym_mixnet_analysis_results.md`

**Thesis Content**:
- Unsupervised clustering analysis of NYM mixnet traffic
- Privacy preservation effectiveness assessment
- Cluster stability and quality metrics
- Proxy classification for separability analysis

**Repository Files**:
```
nym_mixnet_analysis/
├── scripts/
│   ├── nym_pipeline.py                # Core clustering algorithms
│   └── run_nym_unsup.py              # Main execution script
├── nym_unsup/
│   ├── data/
│   │   ├── raw/
│   │   │   └── NYM_Mixnet_IoT_matadata.pcap
│   │   └── processed/
│   │       ├── nym_mixnet_IoT_metadata_subset.csv
│   │       └── nym_metadata.csv
│   ├── requirements.txt               # Dependencies
│   └── README.md                      # Analysis documentation
├── results/                           # Analysis results
│   ├── nym_mixnet_analysis_results.md # Clean results summary
│   ├── comparison_summary.md          # Baseline vs NYM comparison
│   └── tables/                        # Performance tables
│       ├── clustering_summary.csv
│       ├── proxy_classification_results.csv
│       └── confusion_matrix.csv
└── README.md                          # Analysis documentation
```

### Chapter 6: Conclusions and Future Work

**Repository Location**: `thesis_chapters/`

**Key Components**:
- **Conclusions**: `thesis_chapters/chapter6_conclusions.tex`
- **Source Code**: `thesis_chapters/chapter_source_code.tex`
- **Analysis Document**: `thesis_chapters/nym_analysis_document.tex`

**Thesis Content**:
- Comparison with state-of-the-art
- Lessons learned from the research
- Security assessment and residual threats
- Future research perspectives

## Data References

### Raw Data (PCAP Files)
- **IoT Baseline**: Available in `iot_baseline_analysis/data/raw/` - Original IoT device traffic captures
- **NYM Mixnet**: Available in `nym_mixnet_analysis/nym_unsup/data/raw/` - NYM mixnet traffic captures from Raspberry Pi 5 entry gateway

### Processed Data (CSV Files)
- **IoT Baseline**: Available in `iot_baseline_analysis/data/processed/` - Extracted flow-level features
- **NYM Mixnet**: `nym_mixnet_analysis/nym_unsup/data/processed/nym_mixnet_IoT_metadata_subset.csv` - Extracted flow-level features
- **Features**: 14 flow-level metadata features
- **Samples**: 1,642 network flows

### Data Processing Pipeline
1. **PCAP Capture**: Network traffic captured using Wireshark/tcpdump
2. **Flow Extraction**: CICFlowMeter used to extract flow-level metadata
3. **Feature Engineering**: 14 specific features selected for analysis
4. **Data Cleaning**: Missing values and outliers handled appropriately

## Results References

### Baseline Analysis Results
- **Best Model**: Random Forest (98.79% accuracy)
- **Feature Set**: 14 flow-level metadata features
- **Evaluation**: Accuracy, precision, recall, F1-score
- **Location**: Available in `nym_mixnet_analysis/results/tables/baseline_classification_results.csv`

### NYM Mixnet Results
- **Best Clustering**: HDBSCAN (10 clusters, 25.58% noise)
- **Quality Metrics**: Silhouette 0.906, Davies-Bouldin 0.199
- **Stability**: Bootstrap ARI 0.775
- **Location**: `nym_mixnet_analysis/results/nym_mixnet_analysis_results.md`

## Code References

### Core Algorithms
- **Clustering**: HDBSCAN, GMM-BIC, Spectral Clustering
- **Classification**: Random Forest, SVM, Decision Tree, KNN
- **Evaluation**: Silhouette, Davies-Bouldin, Calinski-Harabasz

### Implementation Files
- **Pipeline**: `nym_mixnet_analysis/scripts/nym_pipeline.py`
- **Execution**: `nym_mixnet_analysis/scripts/run_nym_unsup.py`
- **Notebook**: `iot_baseline_analysis/notebooks/1.IoT_Fingerprint_ML_Evaluation.ipynb`

## Citation Information

### Repository Citation
```bibtex
@software{iot_nym_thesis_2025,
  title={IoT Devices Fingerprinting based on Metadata Classification, Threat Models and Countermeasures: Thesis Repository},
  author={Muhammad Siddique},
  year={2025},
  url={https://github.com/galaxyonymous/IoT-NYM-Thesis-Repository},
  note={Complete implementation for thesis research}
}
```

### Thesis Citation
```bibtex
@thesis{siddique2025,
  title={IoT Devices Fingerprinting based on Metadata Classification, Threat Models and Countermeasures},
  author={Muhammad Siddique},
  year={2025},
  school={ULB Brussels},
  note={Repository: https://github.com/galaxyonymous/IoT-NYM-Thesis-Repository}
}
```

## Repository Structure for Thesis

```
IoT_NYM_Thesis_Repository/
├── README.md                          # Main repository overview
├── THESIS_REFERENCE.md                # This document
├── requirements.txt                   # Python dependencies
├── LICENSE                           # MIT License
├── .gitignore                        # Git ignore rules
├── iot_baseline_analysis/            # Chapter 4 implementation
│   ├── notebooks/
│   │   └── 1.IoT_Fingerprint_ML_Evaluation.ipynb
│   ├── data/
│   │   ├── raw/                      # PCAP files
│   │   └── processed/                # CSV files
│   └── figures/                      # Generated plots (14 files)
├── nym_mixnet_analysis/              # Chapter 5 implementation
│   ├── scripts/
│   │   ├── nym_pipeline.py
│   │   └── run_nym_unsup.py
│   ├── nym_unsup/
│   │   ├── data/
│   │   │   ├── raw/                  # PCAP files
│   │   │   └── processed/            # CSV files
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── results/                      # Analysis results
│   │   ├── nym_mixnet_analysis_results.md
│   │   ├── comparison_summary.md
│   │   └── tables/
│   │       ├── clustering_summary.csv
│   │       ├── proxy_classification_results.csv
│   │       └── confusion_matrix.csv
│   └── README.md
└── documentation/                    # Setup and usage guides
    ├── setup/
    │   └── INSTALLATION.md
    └── usage/
        └── USAGE_GUIDE.md
```

## Usage in Thesis

### In-Text References
- "The complete implementation is available in the thesis repository (Siddique, 2025)"
- "Source code and analysis scripts can be found at: https://github.com/galaxyonymous/IoT-NYM-Thesis-Repository"

### Figure and Table References
- "Table 4.1: Baseline classification results (see repository: `iot_baseline_analysis/notebooks/1.IoT_Fingerprint_ML_Evaluation.ipynb`)"
- "Figure 5.2: Clustering performance comparison (see repository: `nym_mixnet_analysis/results/nym_mixnet_analysis_results.md`)"

### Code References
- "Algorithm 1: HDBSCAN clustering implementation (see repository: `nym_mixnet_analysis/scripts/nym_pipeline.py`)"
- "Listing 4.1: IoT baseline analysis (see repository: `iot_baseline_analysis/notebooks/1.IoT_Fingerprint_ML_Evaluation.ipynb`)"

### Data References
- "Raw network traffic captures (see repository: `nym_mixnet_analysis/nym_unsup/data/raw/`)"
- "Processed flow metadata (see repository: `nym_mixnet_analysis/nym_unsup/data/processed/`)"

## Maintenance

### Version Control
- All code is version controlled with Git
- Tagged releases correspond to thesis versions
- Commit history tracks development progress

### Documentation
- Comprehensive README files in each directory
- Installation and usage guides
- Code comments and docstrings
- LaTeX documentation in thesis_chapters/

### Reproducibility
- Fixed random seeds for reproducible results
- Exact dependency versions in requirements.txt
- Complete data processing pipelines
- Automated analysis scripts

## Contact

For questions about the repository or thesis implementation:
- **Author**: Muhammad Siddique
- **Email**: muhammad.siddique@ulb.be
- **GitHub**: [@galaxyonymous](https://github.com/galaxyonymous)
- **University**: ULB Brussels
- **Repository**: https://github.com/galaxyonymous/IoT-NYM-Thesis-Repository
