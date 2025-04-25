# mlflow_test/Dockerfile
FROM python:3.10-slim

# Install MLflow
RUN pip install mlflow

# Expose port
EXPOSE 5000

# Default command
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
