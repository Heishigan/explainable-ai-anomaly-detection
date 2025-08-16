# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Explainable AI project for cybersecurity anomaly detection using the UNSW-NB15 network intrusion dataset. The project focuses on building interpretable machine learning models to detect network attacks and provide explanations for predictions.

## Development Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter Lab for notebook development
jupyter lab

# Navigate to notebooks directory for analysis
cd notebooks
```

## Dataset Architecture

The project uses the UNSW-NB15 dataset with the following structure:

- **Training set**: 175,341 records with 45 features
- **Testing set**: 82,332 records with 45 features 
- **Target variable**: `label` (0=normal, 1=attack)
- **Attack categories**: 9 types including DoS, Exploits, Fuzzers, Generic, Reconnaissance, Analysis, Backdoors, Shellcode, Worms

### Key Features
- **Network flow features**: duration, packet counts, byte counts, protocol info
- **Statistical features**: jitter, load, window sizes, TCP metrics
- **Behavioral features**: connection counts, service patterns, timing patterns
- **Binary indicators**: FTP login status, small IP/port patterns

## Data Files

- `data/raw/NUSW-NB15_features.csv`: Feature descriptions and metadata
- `data/raw/UNSW_NB15_training-set.csv`: Pre-split training data
- `data/raw/UNSW_NB15_testing-set.csv`: Pre-split testing data
- `data/raw/The UNSW-NB15 description.pdf`: Dataset documentation

## Notebook Structure

- `notebooks/01_EDA.ipynb`: Exploratory data analysis with class distribution, correlation analysis, and feature statistics
- `results/eda/eda_summary.json`: EDA results summary for reference

## ML Pipeline Considerations

### Data Preprocessing Requirements
- **Categorical encoding**: Protocol, service, state, attack_cat columns need encoding
- **Feature scaling**: Wide range of numerical values (bytes, rates, counts) require normalization
- **Class imbalance**: 68% attack vs 32% normal traffic - consider sampling strategies
- **High-dimensional**: 45 features may benefit from dimensionality reduction or feature selection

### Explainability Framework
The project uses explainability libraries:
- **SHAP**: For model-agnostic explanations and feature importance
- **LIME**: For local interpretable model explanations

### Expected Model Types
Based on dependencies, the pipeline likely supports:
- **Deep learning**: TensorFlow/Keras for neural networks
- **Traditional ML**: scikit-learn for ensemble methods, SVM, etc.
- **Preprocessing**: pandas, numpy for data manipulation

## Development Notes

- The `src/` directory is currently empty - implementation code should be organized here
- Results are stored in `results/` with subdirectories by analysis type
- Use the pre-split train/test sets to ensure reproducible comparisons
- Feature descriptions in `NUSW-NB15_features.csv` are essential for understanding model explanations
- Windows environment setup - ensure proper path handling for cross-platform compatibility