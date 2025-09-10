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

## Project Implementation Status

The project has been successfully implemented in three phases:

### Phase 1: Core ML Foundation ✅
- **Model Training Pipeline**: `src/models/model_trainer.py` - Complete training pipeline for multiple ML models
- **Model Implementations**: `src/models/` directory with base classes and specific detectors:
  - Random Forest, XGBoost, Logistic Regression
  - Multi-Layer Perceptron (MLP), Gradient Boosting, SVM
- **Data Processing**: `src/data/` with preprocessing and feature engineering
- **Configuration Management**: `src/config/` for centralized settings

### Phase 2: Explainability Framework ✅  
- **SHAP Integration**: `src/explainability/shap_explainer.py` - Advanced SHAP explanations with cybersecurity-specific feature categorization
- **LIME Integration**: `src/explainability/lime_explainer.py` - Local interpretable explanations
- **Explanation Aggregator**: `src/explainability/explanation_aggregator.py` - Unified explanation interface
- **Real-time Explainer**: `src/explainability/realtime_explainer.py` - Live prediction explanations

### Phase 3: Web Dashboard ✅
- **Interactive Dashboard**: `src/explainability/dashboard_interface.py` - Real-time web interface with:
  - Live anomaly detection with explanations
  - Historical attack analytics and trends
  - Interactive feature importance visualization
  - Model performance monitoring
- **FastAPI Backend**: `web_dashboard.py` and `run_web_dashboard.py` - RESTful API with WebSocket support
- **Web Frontend**: `web/` directory with modern HTML5/CSS3/JavaScript interface

## Quick Start Commands

```bash
# Train models (fast training for development)
python train_models_fast.py

# Run web dashboard
python run_web_dashboard.py

# Test individual components
python test_phase1.py        # Test ML pipeline
python test_explainability.py  # Test explanations
python test_web_dashboard.py   # Test dashboard
```

## Web Dashboard Features

- **Real-time Detection**: Live anomaly detection with instant SHAP explanations
- **Attack Analytics**: Comprehensive analysis of detected attacks by category
- **Time-series Visualization**: Adaptive time-series charts with configurable intervals
- **Feature Importance**: Interactive visualization of model decision factors
- **Model Comparison**: Side-by-side model performance metrics

## Development Notes

- **Modular Architecture**: Clean separation between data processing, models, explainability, and web interface
- **Results Storage**: `results/` contains trained models, preprocessing artifacts, and explanations
- **Reproducible Pipeline**: Pre-split datasets ensure consistent evaluation
- **Windows Compatibility**: Proper path handling for cross-platform development
- **Performance Optimized**: Fast training pipeline for development iterations
