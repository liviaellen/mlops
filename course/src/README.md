# ML Workflows

This directory contains two Metaflow flows for machine learning model training and inference.

## Files

- `trainingflow.py`: Flow for training and evaluating machine learning models
- `scoringflow.py`: Flow for making predictions using trained models

## Requirements

All required dependencies are listed in the root `requirements.txt` file.

## Usage

### Training Flow

To run the training flow:

```bash
python src/trainingflow.py run
```

Parameters can be customized:

```bash
python src/trainingflow.py run --test_size 0.3 --n_estimators 200 --cv_folds 5
```

### Scoring Flow

To run the scoring flow:

```bash
python src/scoringflow.py run --input_data "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0"
```

## MLFlow Integration

Both flows integrate with MLFlow for experiment tracking and model registry. Make sure MLFlow is running locally:

```bash
mlflow server --host 0.0.0.0 --port 5001
```

## Notes

- The training flow includes cross-validation and multiple evaluation metrics
- The scoring flow provides both class predictions and probability estimates
- Both flows include visualization capabilities through Metaflow cards
