# MLOps Course Labs

This repository contains the labs and demos for the MLOps course. The repository is structured to support machine learning workflows including data management, model training, and experimentation.

## Project Structure

- `course/Labs/`: Jupyter notebooks for analysis and experimentation
- `data/`: Data files and datasets
- `models/`: Trained models and model artifacts
- `src/`: Source code for data preprocessing and model training

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- mlflow==2.15.1
- numpy==1.26.4
- pandas==2.2.2
- scikit-learn==1.5.1
- dvc==3.50.2
- dvc-s3==3.50.2
