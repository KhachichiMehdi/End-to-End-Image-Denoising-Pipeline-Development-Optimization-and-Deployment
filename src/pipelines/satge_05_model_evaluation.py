from src.config.configurtion import Configuration
from src.components.model_evaluation import ModelEvaluation
from src.utils.logger import logging
from pathlib import Path
STAGE_NAME  = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self) -> None:
        pass

    def main(self):

        config = Configuration(Path("config\config.yaml"),Path("params.yaml"))
        get_config_data = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(get_config_data)
        model_evaluation.initiate_model_evaluation()
        
if __name__ == "__main__":
    
    try:
        logging.info(f" >>>> stage {STAGE_NAME} <<<< started !")
        obj = ModelEvaluationPipeline()
        obj.main()
        logging.info(f" >>>> stage {STAGE_NAME} <<<< Completed ! \n\n x==================x")
        
    except Exception as e:
        logging.exception(e)
        raise e
    