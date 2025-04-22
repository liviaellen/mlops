from metaflow import FlowSpec, step, Parameter, card
import numpy as np
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import time
import matplotlib.pyplot as plt
import seaborn as sns

class ScoringFlow(FlowSpec):
    """
    Inference flow for making predictions using a trained model.
    This flow handles data preprocessing, model loading, and prediction generation
    using a model registered in MLFlow.
    """

    # Input parameter for prediction data
    input_data = Parameter('input_data',
                         help='Comma-separated feature values for prediction',
                         required=True)

    @step
    def start(self):
        """
        Data preparation and model loading step.
        Processes input data and loads the latest trained model from MLFlow.
        """
        try:
            # Input data validation and preprocessing
            features = [float(x.strip()) for x in self.input_data.split(',')]
            if len(features) != 20:  # Match the number of features in training
                raise ValueError(f"Expected 20 features, got {len(features)}")

            self.X = np.array(features).reshape(1, -1)

            # MLFlow connection with retry mechanism
            max_retries = 3
            retry_delay = 5

            for attempt in range(max_retries):
                try:
                    mlflow.set_tracking_uri("http://localhost:5001")
                    self.model = mlflow.sklearn.load_model("models:/metaflow-rf-model/latest")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise Exception(f"Model loading failed after {max_retries} attempts: {str(e)}")
                    print(f"Model loading attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)

            # Feature scaling (using the same scaler as in training)
            scaler = StandardScaler()
            self.X_scaled = scaler.fit_transform(self.X)

            self.next(self.predict)

        except ValueError as ve:
            print(f"Input data validation error: {str(ve)}")
            raise
        except Exception as e:
            print(f"Error in data preparation: {str(e)}")
            raise

    @step
    def predict(self):
        """
        Prediction generation step.
        Generates class predictions and probability estimates.
        """
        try:
            # Generate predictions
            self.prediction = self.model.predict(self.X_scaled)[0]
            self.probability = self.model.predict_proba(self.X_scaled)[0]

            # Create probability visualization
            plt.figure(figsize=(8, 6))
            sns.barplot(x=[f'Class {i}' for i in range(len(self.probability))],
                       y=self.probability)
            plt.title('Class Probability Distribution')
            plt.ylabel('Probability')
            plt.ylim(0, 1)
            self.probability_plot = plt.gcf()

            self.next(self.end)

        except Exception as e:
            print(f"Error in prediction generation: {str(e)}")
            raise

    @card
    @step
    def end(self):
        """
        Results presentation step.
        Displays prediction results in a clear format with visualizations.
        """
        from metaflow.cards import Markdown, Image

        # Create results card
        results_md = f"""
        # Prediction Results

        ## Input Features
        {self.X[0]}

        ## Prediction
        Predicted Class: {self.prediction}

        ## Class Probabilities
        {dict(zip([f'Class {i}' for i in range(len(self.probability))], self.probability))}
        """

        # Save probability plot
        self.probability_plot.savefig('class_probabilities.png')

        # Add content to card
        self.card.append(Markdown(results_md))
        self.card.append(Image('class_probabilities.png'))

        print("\nPrediction Results:")
        print(f"Input Features: {self.X[0]}")
        print(f"Predicted Class: {self.prediction}")
        print("\nClass Probabilities:")
        for i, prob in enumerate(self.probability):
            print(f"Class {i}: {prob:.4f}")

if __name__ == '__main__':
    ScoringFlow()
