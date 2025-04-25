ML Workflows with GCP Integration

This directory contains two Metaflow flows for machine learning model training and inference, designed to run on Google Cloud Platform (GCP) using Kubernetes.

## Files

- `trainingflowgcp.py`: Flow for training a Random Forest classifier using the wine dataset
- `scoringflowgcp.py`: Flow for making predictions using the trained model

## Requirements

All required dependencies are listed in the root `requirements.txt` file. The main dependencies are:
- Python 3.9.16
- scikit-learn 1.2.2
- pandas 1.5.3
- metaflow

## Environment Setup

1. First, activate the correct Python environment:

```bash
pyenv activate mlops
```

2. Install dependencies:

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Verify Metaflow installation
metaflow --version
```

## Usage

### Training Flow

To run the training flow with Kubernetes:

```bash
python trainingflowgcp.py --environment=conda run --with kubernetes
```

The training flow:
- Loads the wine dataset from scikit-learn
- Splits data into training (80%) and test (20%) sets
- Trains a Random Forest classifier with 100 trees
- Reports model accuracy on the test set
- Runs on Kubernetes with retry and timeout capabilities

### Scoring Flow

To run the scoring flow with Kubernetes:

```bash
python scoringflowgcp.py --environment=conda run --with kubernetes
```

The scoring flow:
- Loads the first 5 samples from the wine dataset
- Trains a Random Forest model (50 trees) on the remaining data
- Makes predictions and probability estimates for the 5 samples
- Displays detailed prediction results
- Runs on Kubernetes with retry and timeout capabilities

## Kubernetes Integration

Both flows use Kubernetes for execution and include:
- Retry mechanism (3 attempts)
- 10-minute timeout protection
- Error catching
- Conda environment specification

## Notes

- The training flow includes:
  - Random Forest classifier with 100 trees
  - 80/20 train-test split
  - Accuracy evaluation
  - Kubernetes execution

- The scoring flow provides:
  - Class predictions
  - Probability estimates for each class
  - Detailed output for 5 samples
  - Kubernetes execution

## Troubleshooting

If you encounter any issues:

1. Verify you're in the correct environment:
```bash
pyenv versions  # Check available environments
pyenv activate mlops  # Activate the correct environment
```

2. Check Metaflow installation:
```bash
metaflow --version
```

3. Verify Kubernetes configuration:
```bash
kubectl config current-context  # Should show your GCP cluster
```

4. Check logs if a flow fails:
```bash
python src/trainingflowgcp.py logs
python src/scoringflowgcp.py logs
```
