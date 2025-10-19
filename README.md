MLops week4 GA
# IRIS ML Pipeline - MLOps CI/CD Project

[![CI Pipeline - Dev Branch](https://github.com/Asadityasingh/IRIS-ML-pipeline/actions/workflows/ci-dev.yml/badge.svg)](https://github.com/Asadityasingh/IRIS-ML-pipeline/actions/workflows/ci-dev.yml)
[![CI Pipeline - Main Branch](https://github.com/Asadityasingh/IRIS-ML-pipeline/actions/workflows/ci-main.yml/badge.svg)](https://github.com/Asadityasingh/IRIS-ML-pipeline/actions/workflows/ci-main.yml)

A complete MLOps pipeline for IRIS flower classification with automated testing, CI/CD, and model versioning using DVC.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [Installation](#installation)
- [Usage](#usage)
- [CI/CD Pipeline](#cicd-pipeline)
- [Testing](#testing)
- [Model Performance](#model-performance)
- [Contributing](#contributing)

## 🎯 Project Overview

This project implements a complete MLOps workflow for training and deploying an IRIS flower classification model using Random Forest. It includes:

- **Automated Testing**: 13 comprehensive unit tests for data validation and model evaluation
- **CI/CD Integration**: GitHub Actions workflows for continuous integration and deployment
- **Model Versioning**: DVC (Data Version Control) for tracking model artifacts
- **Automated Reporting**: CML (Continuous Machine Learning) for generating performance reports

## 📁 Project Structure

```

IRIS-ML-pipeline/
├── .dvc/                           \# DVC configuration directory
│   ├── config                      \# DVC remote storage configuration
│   └── .gitignore                  \# DVC internal files to ignore
│
├── .github/                        \# GitHub-specific configurations
│   └── workflows/                  \# CI/CD workflow definitions
│       ├── ci-dev.yml              \# CI pipeline for dev branch
│       └── ci-main.yml             \# CI pipeline for main branch
│
├── models/                         \# Model artifacts directory
│   ├── iris_model.pkl              \# Trained Random Forest model (ignored by Git)
│   ├── iris_model.pkl.dvc          \# DVC metadata for model tracking
│   └── .gitignore                  \# Ignores actual model file
│
├── src/                            \# Source code directory
│   ├── __init__.py                 \# Python package initializer
│   ├── train.py                    \# Model training script
│   └── evaluate.py                 \# Model evaluation script
│
├── tests/                          \# Test suite directory
│   ├── __init__.py                 \# Test package initializer
│   ├── test_data_validation.py    \# Data validation unit tests (6 tests)
│   └── test_model_evaluation.py   \# Model evaluation unit tests (7 tests)
│
├── metrics/                        \# Generated metrics directory
│   └── evaluation.json             \# Model performance metrics
│
├── data/                           \# Data directory (optional)
│   └── iris.csv                    \# IRIS dataset
│
├── .gitignore                      \# Git ignore patterns
├── .dvcignore                      \# DVC ignore patterns
├── requirements.txt                \# Python dependencies
└── README.md                       \# This file

```

## 📄 File Descriptions

### Core Configuration Files

#### **requirements.txt**
**Purpose**: Defines all Python package dependencies with version constraints
```


# Key dependencies:

- numpy>=1.26.0,<2.0.0    \# Numerical computing (with NumPy 2.0 compatibility fix)
- pandas>=2.2.0            \# Data manipulation
- scikit-learn>=1.5.0      \# Machine learning algorithms
- pytest>=7.4.0            \# Testing framework
- dvc>=3.55.0              \# Data version control
- joblib>=1.3.2            \# Model serialization

```
**Usage**: Install with `pip install -r requirements.txt`

#### **.gitignore**
**Purpose**: Specifies files and directories Git should ignore
```


# Ignores:

- Python cache files (__pycache__, *.pyc)
- Virtual environments (venv/, env/)
- Model files (handled by DVC)
- IDE configurations (.vscode/, .idea/)
- OS-specific files (.DS_Store)

```

#### **.dvcignore**
**Purpose**: Specifies files DVC should ignore when tracking
```


# Similar to .gitignore but for DVC operations

```

### DVC (Data Version Control) Files

#### **.dvc/config**
**Purpose**: Stores DVC remote storage configuration
```

[core]
remote = myremote
['remote "myremote"']
url = /home/aditya/dvc-storage  \# Local DVC remote path

```
**Utility**: 
- Defines where model artifacts are stored
- Enables model versioning and sharing across team
- Can be configured for cloud storage (S3, GCS, Azure Blob)

#### **models/iris_model.pkl.dvc**
**Purpose**: DVC metadata file tracking the actual model
```

outs:

- md5: abc123...          \# Hash of the model file
size: 123456            \# File size in bytes
path: iris_model.pkl    \# Path to actual model

```
**Utility**:
- Tracked by Git (lightweight text file)
- Points to actual model stored in DVC remote
- Enables model version control without bloating Git repository

### Source Code Files

#### **src/train.py**
**Purpose**: Main training script for the IRIS classification model

**Key Functions**:
```

def train_model():
\# Loads IRIS dataset
\# Splits into train/test (80/20)
\# Trains Random Forest classifier (100 estimators)
\# Saves model to models/iris_model.pkl
\# Generates training metrics

```

**Usage**:
```

python src/train.py

# Output: Model trained with accuracy: 1.0000

```

**Utility**:
- Creates reproducible model artifacts
- Generates training metrics for tracking
- Used by CI/CD pipeline for automated training

#### **src/evaluate.py**
**Purpose**: Evaluates trained model and generates performance metrics

**Key Functions**:
```

def evaluate_model(model_path='models/iris_model.pkl'):
\# Loads trained model
\# Runs predictions on test set
\# Calculates accuracy, precision, recall, F1-score
\# Saves metrics to metrics/evaluation.json

```

**Generated Metrics**:
```

{
"accuracy": 1.0000,
"precision": 1.0000,
"recall": 1.0000,
"f1_score": 1.0000
}

```

**Usage**:
```

python src/evaluate.py

# Output: Evaluation completed with metrics

```

**Utility**:
- Provides standardized evaluation metrics
- Used by CI/CD for automated reporting
- Enables performance tracking across versions

### Test Files

#### **tests/test_data_validation.py**
**Purpose**: Validates IRIS dataset integrity and quality

**Test Coverage** (6 tests):
1. **test_data_shape**: Verifies dataset has 150 samples and 4 features
2. **test_data_types**: Ensures all features are numeric
3. **test_no_missing_values**: Checks for missing data
4. **test_feature_ranges**: Validates feature values are reasonable
5. **test_target_classes**: Confirms 3 classes (0, 1, 2)
6. **test_class_distribution**: Verifies balanced classes (50 samples each)

**Usage**:
```

pytest tests/test_data_validation.py -v

```

**Utility**:
- Catches data quality issues early
- Ensures training data consistency
- Prevents silent failures from corrupted data

#### **tests/test_model_evaluation.py**
**Purpose**: Validates model behavior and performance

**Test Coverage** (7 tests):
1. **test_model_accuracy**: Ensures accuracy ≥ 85%
2. **test_model_predictions_shape**: Validates prediction dimensions
3. **test_model_predictions_range**: Checks predictions are in valid classes
4. **test_model_proba_output**: Validates probability outputs sum to 1.0
5. **test_model_deterministic**: Ensures consistent predictions
6. **test_single_sample_prediction**: Tests single-sample inference
7. **test_model_load**: Verifies model can be loaded

**Usage**:
```

pytest tests/test_model_evaluation.py -v

```

**Utility**:
- Prevents model regression
- Validates model meets performance thresholds
- Ensures production-ready behavior

### CI/CD Workflow Files

#### **.github/workflows/ci-dev.yml**
**Purpose**: Automated CI pipeline for dev branch and pull requests

**Triggers**:
- Push to `dev` branch
- Pull requests to `main` branch

**Pipeline Steps**:
```

1. Checkout code
2. Set up Python 3.9
3. Install dependencies
4. Train model (fresh training for reproducibility)
5. Run data validation tests (6 tests)
6. Run model evaluation tests (7 tests)
7. Generate test coverage report
8. Evaluate model and generate metrics
9. Setup CML (Continuous Machine Learning)
10. Create CML report as PR comment
```

**CML Report Includes**:
- Model performance metrics table
- Test results summary
- Data validation status
- Timestamp and branch info

**Utility**:
- Automated quality gates for pull requests
- Prevents broken code from merging
- Provides immediate feedback to developers
- Generates visual reports on PR

#### **.github/workflows/ci-main.yml**
**Purpose**: Automated CI pipeline for main branch (production)

**Triggers**:
- Push/merge to `main` branch

**Pipeline Steps**:
Same as dev pipeline, but with production-focused reporting

**Utility**:
- Validates production deployments
- Creates production metrics reports
- Ensures main branch always passes all tests
- Documents model performance in production

### Generated Files

#### **metrics/evaluation.json**
**Purpose**: Stores model performance metrics in JSON format

**Content**:
```

{
"accuracy": 1.0,
"precision": 1.0,
"recall": 1.0,
"f1_score": 1.0,
"test_samples": 30
}

```

**Utility**:
- Machine-readable metrics for automation
- Used by CML for report generation
- Enables performance tracking over time
- Can be integrated with MLflow, Weights & Biases, etc.

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- Git
- DVC (Data Version Control)

### Setup Steps

```


# 1. Clone the repository

git clone https://github.com/Asadityasingh/IRIS-ML-pipeline.git
cd IRIS-ML-pipeline

# 2. Create virtual environment

python3 -m venv venv
source venv/bin/activate  \# On Windows: venv\Scripts\activate

# 3. Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

# 4. Pull model from DVC (if available)

dvc pull

# Or train fresh model

python src/train.py

```

## 💻 Usage

### Training the Model

```


# Train Random Forest model on IRIS dataset

python src/train.py

# Output:

# Loading IRIS dataset...

# Dataset shape: (150, 4)

# Training Random Forest model...

# Model saved to: models/iris_model.pkl

# ✅ Model trained successfully!

# Accuracy: 1.0000

```

### Evaluating the Model

```


# Generate evaluation metrics

python src/evaluate.py

# Output:

# Loading model from: models/iris_model.pkl

# Evaluating model...

# ✅ Evaluation completed!

# Accuracy: 1.0000

# Precision: 1.0000

# Recall: 1.0000

# F1 Score: 1.0000

```

### Running Tests

```


# Run all tests

pytest tests/ -v

# Run specific test file

pytest tests/test_data_validation.py -v

# Run with coverage report

pytest tests/ --cov=src --cov-report=html

```

### Using DVC

```


# Track new model version

dvc add models/iris_model.pkl
git add models/iris_model.pkl.dvc
git commit -m "Update model version"

# Push model to DVC remote

dvc push

# Pull model from DVC remote

dvc pull

# Check DVC status

dvc status

```

## 🔄 CI/CD Pipeline

### Branch Strategy

- **main**: Production-ready code
- **dev**: Development branch for new features

### Workflow

```

1. Develop on dev branch
↓
2. Create Pull Request to main
↓
3. CI runs automatically:
    - Installs dependencies
    - Trains model
    - Runs 13 unit tests
    - Generates metrics
↓
4. CML posts report as PR comment
↓
5. Review and merge if all checks pass
↓
6. Main branch CI runs
↓
7. Production deployment ready
```

### Viewing CI/CD Results

1. Go to **Actions** tab on GitHub
2. Click on workflow run
3. View detailed logs for each step
4. Check CML report on PR or commit

## 🧪 Testing

### Test Coverage

| Category | Tests | Coverage |
|----------|-------|----------|
| Data Validation | 6 | 100% |
| Model Evaluation | 7 | 100% |
| **Total** | **13** | **100%** |

### Data Validation Tests

✅ Dataset shape validation (150 samples, 4 features)  
✅ Data types verification (all numeric)  
✅ Missing values check (no missing values)  
✅ Feature ranges validation (0-10 range)  
✅ Target classes verification (3 classes: 0, 1, 2)  
✅ Class distribution check (balanced: 50 each)

### Model Evaluation Tests

✅ Model accuracy threshold (≥85%)  
✅ Prediction shape validation  
✅ Prediction range check (valid classes)  
✅ Probability output validation (sum=1.0)  
✅ Model determinism test  
✅ Single sample prediction  
✅ Model loading verification

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1-Score | 100% |

**Algorithm**: Random Forest (100 estimators)  
**Dataset**: IRIS (150 samples, 3 classes)  
**Train/Test Split**: 80/20 (120/30 samples)

## 🤝 Contributing

### Development Workflow

```


# 1. Create feature branch from dev

git checkout dev
git pull origin dev
git checkout -b feature/your-feature-name

# 2. Make changes and test locally

python src/train.py
pytest tests/ -v

# 3. Commit and push

git add .
git commit -m "Add your feature"
git push origin feature/your-feature-name

# 4. Create Pull Request to dev branch

# 5. Wait for CI to pass

# 6. Request review and merge

```

### Adding New Tests

```


# tests/test_your_feature.py

import pytest

class TestYourFeature:
def test_something(self):
\# Your test code
assert True

```

## 📝 License

This project is part of the MLOps Week 4 Graded Assignment.

## 🙏 Acknowledgments

- **Dataset**: IRIS dataset from scikit-learn
- **Tools**: DVC, GitHub Actions, CML, pytest
- **Course**: IIT Madras BS MLOps Program

## 📧 Contact

**Author**: Asadityasingh  
**Repository**: [IRIS-ML-pipeline](https://github.com/Asadityasingh/IRIS-ML-pipeline)

---

## 🔍 Quick Reference

### Common Commands

```


# Development

python src/train.py              \# Train model
python src/evaluate.py           \# Evaluate model
pytest tests/ -v                 \# Run tests

# DVC

dvc add models/iris_model.pkl   \# Track with DVC
dvc push                         \# Push to remote
dvc pull                         \# Pull from remote

# Git

git checkout dev                 \# Switch to dev
git pull origin dev              \# Update dev
git push origin dev              \# Push changes

```

### Troubleshooting

**Issue**: NumPy compatibility error  
**Solution**: Ensure `numpy<2.0.0` in requirements.txt

**Issue**: Model file not found  
**Solution**: Run `dvc pull` or `python src/train.py`

**Issue**: Tests failing  
**Solution**: Check model exists and is trained properly

---

**Last Updated**: October 19, 2025  
**Version**: 1.0.0
```

Now create this README file in your project:

```bash
# Save the README
cat > README.md << 'EOF'
[paste the entire README content above]
EOF

# Add to git
git add README.md
git commit -m "Add comprehensive README with file descriptions"
git push origin main
```

This README provides:

- Complete project overview
- Detailed file-by-file descriptions
- Installation and usage instructions
- CI/CD pipeline documentation
- Testing documentation
- Contributing guidelines
- Quick reference commands
