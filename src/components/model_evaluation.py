import tensorflow as tf
from pathlib import Path
from dataclasses import dataclass
from src.utils.exception import CustomException
from ..utils.logger import logging
from ..utils.common import load_model
from src.entity.config_entity import ModelEvaluationConfig
import matplotlib.pyplot as plt
import sys
import numpy as np
import json
import mlflow
import mlflow.tensorflow

@dataclass
class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig) -> None:
        self.config = config
        self.model = None
        

    def evaluate_model(self, model, test_data, x_test_noisy):
        try:
            logging.info("Evaluating the model...")
            loss = model.evaluate(x_test_noisy, test_data, verbose=0)
            logging.info(f"Test Loss (MSE): {loss}")
            return {"mse": loss}
        except Exception as e:
            logging.error(f"Failed to evaluate the model: {e}")
            raise CustomException(e, sys)

    def initiate_model_evaluation(self):
        """
        Execute the full model evaluation process with enhanced MLflow tracking.
        """
        try:
            # Start MLflow run
            with mlflow.start_run():
                self.model = load_model(path=self.config.path_of_model)
            

                # Log the model under mlflow
                mlflow.tensorflow.log_model(self.model, artifact_path="model")

                # Log model parameters
                mlflow.log_param("model_type", "autoencoder_denosing")
                mlflow.log_param("im_size", self.config.im_size)
                mlflow.log_param("batch_size", self.config.batch_size)
                mlflow.log_param("learning_rate", self.config.base_learning_rate)
                mlflow.log_param("epochs", self.config.num_epochs)

                # Evaluate the model
                report = self.evaluate_model(self.model, self.config.X_test, self.config.x_test_noisy)

                # Log evaluation metrics
                mlflow.log_metrics(report)

                # Save evaluation report to file and log as artifact
                logging.info(f"Saving evaluation report to {self.config.evaluation_report_path}")
                self.save_json(path=self.config.evaluation_report_path, report=report)
                logging.info("Report saved successfully.")
                mlflow.log_artifact(self.config.evaluation_report_path)

                # Log additional artifacts (e.g., model summary, plots)
                self.log_additional_artifacts(self.model)
        except Exception as e:
            logging.error(f"An error occurred during model evaluation: {e}")
            raise CustomException(e, sys)



    def log_additional_artifacts(self, model):
        """
        Log additional artifacts like model summary, sample input/output, and evaluation plots.
        """
        try:
            # Log model summary as text artifact
            model_summary_path = Path(self.config.evaluation_report_path).parent / "model_summary.txt"
            with open(model_summary_path, "w", encoding="utf-8") as f:
                model.summary(print_fn=lambda x: f.write(x + "\n"))
            mlflow.log_artifact(str(model_summary_path))
            logging.info("Model summary logged successfully.")

            # Log sample input/output images
            self.log_sample_images()
        except Exception as e:
            logging.error(f"Failed to log additional artifacts: {e}")
            raise CustomException(e, sys)

    def log_sample_images(self):
        """
        Log sample input and output images to visualize the model's performance.
        """
        try:
            sample_input = self.config.x_test_noisy[:5]
            sample_output = self.config.X_test[:5]
            predictions = self.model.predict(sample_input)
            fig, axes = plt.subplots(3, 5, figsize=(15, 9))
            for i in range(5):
                input_shape = sample_input[i].shape
                if len(input_shape) == 3:
                    # Handle RGB images
                    axes[0, i].imshow(sample_input[i].reshape(input_shape))
                    axes[1, i].imshow(predictions[i].reshape(input_shape))
                    axes[2, i].imshow(sample_output[i].reshape(input_shape))
                else:
                    # Handle grayscale images
                    axes[0, i].imshow(sample_input[i].reshape(self.config.im_size), cmap='gray')
                    axes[1, i].imshow(predictions[i].reshape(self.config.im_size), cmap='gray')
                    axes[2, i].imshow(sample_output[i].reshape(self.config.im_size), cmap='gray')

                axes[0, i].set_title("Noisy Input")
                axes[0, i].axis('off')
                axes[1, i].set_title("Reconstruction")
                axes[1, i].axis('off')
                axes[2, i].set_title("Original")
                axes[2, i].axis('off')

            sample_images_path = Path(self.config.evaluation_report_path).parent / "sample_images.png"
            plt.savefig(sample_images_path)
            plt.close()
            mlflow.log_artifact(str(sample_images_path))
            logging.info("Sample images logged successfully.")
        except Exception as e:
            logging.error(f"Failed to log sample images: {e}")
            raise CustomException(e, sys)

   
    @staticmethod
    def save_json(path: Path, report: dict) -> None:
        try:
            with open(path, "w") as f:
                json.dump(report, f)
        except Exception as e:
            logging.error(f"Failed to save evaluation report: {e}")
            raise CustomException(e, sys)

