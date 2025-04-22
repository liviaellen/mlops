# ML Workflows

This directory contains two Metaflow flows for machine learning model training and inference.

## Files

- `trainingflow.py`: Flow for training and evaluating machine learning models
- `scoringflow.py`: Flow for making predictions using trained models

## Requirements

All required dependencies are listed in the root `requirements.txt` file.

## Environment Setup

1. First, activate the correct Python environment:
```bash
# Deactivate current environment if any
pyenv deactivate

# Activate the mlops environment
pyenv activate mlops
```

2. Start the MLFlow server:
```bash
mlflow server --host 0.0.0.0 --port 5001
```

## Usage

### Training Flow

To run the training flow:

```bash
# Make sure you're in the mlops environment
python src/trainingflow.py run --test_size 0.3 --n_estimators 200 --cv_folds 5
```

Parameters can be customized:
- `test_size`: Proportion of dataset to include in test split (default: 0.3)
- `n_estimators`: Number of trees in the random forest (default: 200)
- `cv_folds`: Number of cross-validation folds (default: 5)

### Scoring Flow

To run the scoring flow:

```bash
# Make sure you're in the mlops environment
python src/scoringflow.py run --input_data "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0"
```

The input data should be a comma-separated string of 20 numerical values.

## MLFlow Integration

Both flows integrate with MLFlow for experiment tracking and model registry. Important notes:

1. The MLFlow server must be running before executing either flow
2. The server should be running on port 5001
3. Make sure you're in the correct Python environment (`mlops`) when running the MLFlow server

## Notes

- The training flow includes:
  - Cross-validation
  - Multiple evaluation metrics (accuracy, precision, recall, F1 score)
  - Feature importance visualization
  - Model registration in MLFlow

- The scoring flow provides:
  - Class predictions
  - Probability estimates
  - Visualization of class probabilities
  - Results card with detailed output

## Troubleshooting

If you encounter any issues:

1. Verify you're in the correct environment:
```bash
pyenv versions  # Check available environments
pyenv activate mlops  # Activate the correct environment
```

2. Check if MLFlow server is running:
```bash
# Should show MLFlow UI is running
curl http://localhost:5001
```

3. Verify model exists in MLFlow registry:
```bash
# Visit MLFlow UI in browser
http://localhost:5001
```
