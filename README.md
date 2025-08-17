# IoT Devices Fingerprinting based on Metadata Classification, Threat Models and Countermeasures: Thesis Repository

## Overview

This repository contains the complete implementation and analysis for the thesis research on IoT device privacy preservation using NYM mixnet technology on Raspberry Pi 5 entry gateways. The research demonstrates the effectiveness of privacy-preserving networks in obfuscating IoT device fingerprints through traffic pattern analysis.

## Repository Structure

```
IoT_NYM_Thesis_Repository/
├── iot_baseline_analysis/          # Baseline IoT device fingerprinting
│   ├── notebooks/                  # Jupyter notebooks
│   │   └── 1.IoT_Fingerprint_ML_Evaluation.ipynb
│   ├── data/                       # IoT baseline data
│   │   ├── raw/                    # PCAP files
│   │   │   ├── Sample_Metadata_Smart_Plug.pcap
│   │   │   ├── Sample_Metadata_lamp.pcap
│   │   │   └── IoT_Network_Devices_MetaData_RPI_GW.pcap
│   │   └── processed/              # CSV files
│   │       ├── Sample_Metadata_Smart_Plug.csv
│   │       ├── Sample_Metadata_Smart_Lamp.csv
│   │       └── IoT_metadata.csv
│   └── figures/                    # Generated plots (14 files)
│       ├── baseline_scaling_effect_barh.png
│       ├── baseline_learning_curves_single.png
│       ├── baseline_pr_2.png
│       ├── baseline_pr.png
│       ├── baseline_roc.png
│       ├── baseline_learning_curves_2x2.png
│       ├── baseline_cluster_sizes.png
│       ├── baseline_kmeans_silhouette.png
│       ├── baseline_kmeans_elbow.png
│       ├── baseline_perm_importance_svm.png
│       ├── baseline_rf_importance.png
│       ├── baseline_cv_boxplot.png
│       ├── baseline_confusion_grid.png
│       └── baseline-metrics-grouped.png
├── nym_mixnet_analysis/            # NYM mixnet privacy analysis
│   ├── scripts/                    # Core pipeline scripts
│   │   ├── nym_pipeline.py
│   │   └── run_nym_unsup.py
│   ├── nym_unsup/                  # Analysis directory
│   │   ├── data/                   # NYM data files
│   │   │   ├── raw/                # PCAP files
│   │   │   │   └── NYM_Mixnet_IoT_matadata.pcap
│   │   │   └── processed/          # CSV files
│   │   │       ├── nym_mixnet_IoT_metadata_subset.csv
│   │   │       └── nym_metadata.csv
│   │   ├── requirements.txt        # Dependencies
│   │   └── README.md               # Analysis documentation
│   ├── results/                    # Analysis results
│   │   ├── nym_mixnet_analysis_results.md  # Clean results summary
│   │   ├── comparison_summary.md   # Baseline vs NYM comparison
│   │   └── tables/                 # Performance tables and metrics
│   │       ├── clustering_summary.csv
│   │       ├── proxy_classification_results.csv
│   │       └── confusion_matrix.csv
│   └── README.md                   # NYM analysis documentation
├── documentation/                  # Setup and usage guides
│   ├── setup/                      # Installation instructions
│   │   └── INSTALLATION.md
│   └── usage/                      # Usage examples
│       └── USAGE_GUIDE.md
├── README.md                       # This file
├── THESIS_REFERENCE.md             # Thesis mapping guide
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
└── .gitignore                      # Git ignore rules
```

## Key Findings

- **Device Discovery**: Successfully identified 10 distinct device clusters from NYM mixnet traffic (vs. 2 original devices)
- **Privacy Protection**: 25.58% noise rate enhances privacy through traffic obfuscation
- **Classification Accuracy**: 98.79% baseline accuracy (Random Forest) reduced through mixnet implementation
- **Cluster Quality**: Excellent separation (Silhouette: 0.906, Davies-Bouldin: 0.199)

## Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- Jupyter Notebook

### Installation
```bash
# Clone the repository
git clone https://github.com/galaxyonymous/IoT-NYM-Thesis-Repository.git
cd IoT-NYM-Thesis-Repository

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

#### IoT Baseline Analysis
```bash
cd iot_baseline_analysis/notebooks
jupyter notebook 1.IoT_Fingerprint_ML_Evaluation.ipynb
```

#### NYM Mixnet Analysis
```bash
cd nym_mixnet_analysis/scripts
python run_nym_unsup.py
```

## Dataset Information

The analysis uses network flow metadata extracted from PCAP files:

### Raw Data (PCAP Files)
- **IoT Baseline**: Available in `iot_baseline_analysis/data/raw/` - Original IoT device traffic captures
- **NYM Mixnet**: Available in `nym_mixnet_analysis/nym_unsup/data/raw/` - NYM mixnet traffic captures from Raspberry Pi 5 entry gateway

### Processed Data (CSV Files)
- **IoT Baseline**: Available in `iot_baseline_analysis/data/processed/` - Extracted flow-level features
- **NYM Mixnet**: Available in `nym_mixnet_analysis/nym_unsup/data/processed/` - Extracted flow-level features
- **Features**: 14 flow-level metadata features
- **Samples**: 1,642 network flows (NYM mixnet)

### Data Processing Pipeline
1. **PCAP Capture**: Network traffic captured using Wireshark/tcpdump
2. **Flow Extraction**: CICFlowMeter used to extract flow-level metadata
3. **Feature Engineering**: 14 specific features selected for analysis
4. **Data Cleaning**: Missing values and outliers handled appropriately

## Results

### Clustering Performance
| Method | Clusters | Noise Rate | Silhouette | Davies-Bouldin | Stability |
|--------|----------|------------|------------|----------------|-----------|
| HDBSCAN | 10 | 25.58% | 0.906 | 0.199 | 0.775 |
| GMM-BIC | 7 | 0.00% | 0.394 | 1.632 | N/A |
| Spectral | 8 | 0.00% | -0.136 | 1.583 | N/A |

### Baseline Classification Results
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 98.79% | 98.83% | 98.79% | 98.79% |
| SVM (RBF Kernel) | 88.79% | 89.09% | 88.79% | 88.41% |
| Decision Tree | 98.48% | 98.51% | 98.48% | 98.49% |
| K-Nearest Neighbors | 91.82% | 91.77% | 91.82% | 91.77% |

## Thesis Reference

This repository serves as the complete implementation reference for the thesis:

**Title**: IoT Devices Fingerprinting based on Metadata Classification, Threat Models and Countermeasures

**Chapters**:
- Chapter 4: IoT Device Fingerprinting Baseline Analysis
- Chapter 5: NYM Mixnet Privacy Preservation Assessment
- Chapter 6: Conclusions and Future Work

**Key Files**:
- **Baseline Analysis**: `iot_baseline_analysis/notebooks/1.IoT_Fingerprint_ML_Evaluation.ipynb`
- **NYM Analysis**: `nym_mixnet_analysis/scripts/run_nym_unsup.py`
- **Results Summary**: `nym_mixnet_analysis/results/nym_mixnet_analysis_results.md`
- **Thesis Mapping**: `THESIS_REFERENCE.md`

## Citation

If you use this code in your research, please cite:

```bibtex
@thesis{siddique2025,
  title={IoT Devices Fingerprinting based on Metadata Classification, Threat Models and Countermeasures},
  author={Muhammad Siddique},
  year={2025},
  school={ULB Brussels}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Muhammad Siddique
- **Email**: muhammad.siddique@ulb.be
- **GitHub**: [@galaxyonymous](https://github.com/galaxyonymous)
- **University**: ULB Brussels

---

**Note**: This repository contains research code and may require specific datasets or configurations. Please refer to the documentation for detailed setup instructions.
