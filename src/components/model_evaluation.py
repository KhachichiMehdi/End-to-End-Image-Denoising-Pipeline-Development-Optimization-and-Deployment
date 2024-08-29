
import tensorflow as tf
from pathlib import Path
from dataclasses import dataclass
from src.utils.exception import CustomException
from ..utils.logger import logging
from ..utils.common import load_model 
from src.entity.config_entity import ModelEvaluationConfig
from pathlib import Path
import sys
import numpy as np
import json

@dataclass
class ModelEvaluation:


          def __init__(self ,config: ModelEvaluationConfig) ->None :
                    self.config=config



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
                    Exécute le processus complet d'évaluation du modèle.
                     """
                    try:
                              model = load_model(path=self.config.path_of_model)
                              report = self.evaluate_model(model, self.config.X_test, self.config.x_test_noisy)
                              logging.info(f"Saving evaluation report to {self.config.evaluation_report_path}")
                              self.save_json(path= self.config.evaluation_report_path, report= report)
                              logging.info("report saved successfully.")
                    except Exception as e:
                              logging.error(f"An error occurred during model evaluation: {e}")
                              raise CustomException(e, sys)
          @staticmethod
          def save_json(path :Path, report : dict)->None:
                    try:
                              with open(path, "w") as f:
                                        json.dump(report, f)
                    except Exception as e:
                              logging.error(f"Failed to save evaluation report: {e}")
                              raise CustomException(e, sys)


