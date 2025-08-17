# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/IoT-NYM-Thesis-Repository.git
cd IoT-NYM-Thesis-Repository
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import numpy, pandas, sklearn, matplotlib, hdbscan; print('All packages installed successfully!')"
```

## Troubleshooting

### HDBSCAN Installation Issues

If you encounter issues installing HDBSCAN:

```bash
# Try installing with conda instead
conda install -c conda-forge hdbscan

# Or install from wheel
pip install --only-binary=all hdbscan
```

### Jupyter Issues

If Jupyter doesn't start:

```bash
# Install jupyter separately
pip install jupyter notebook

# Or use conda
conda install jupyter notebook
```

## Environment Setup

### For IoT Baseline Analysis

```bash
cd iot_baseline_analysis/notebooks
jupyter notebook
```

### For NYM Mixnet Analysis

```bash
cd nym_mixnet_analysis/scripts
python run_nym_unsup.py
```

## Data Setup

1. Place your dataset files in the `data/raw/` directory
2. Ensure the data files are named correctly:
   - `iot_metadata.csv` for baseline analysis
   - `nym_metadata.csv` for NYM analysis

## Verification

Run the following commands to verify everything is working:

```bash
# Test IoT baseline analysis
cd iot_baseline_analysis/notebooks
python -c "import pandas as pd; print('IoT analysis ready')"

# Test NYM analysis
cd ../../nym_mixnet_analysis/scripts
python -c "import hdbscan; print('NYM analysis ready')"
```
