from src.config.configurtion import Configuration
from src.components.model_training import ModelTraining
from src.components.model_callbacks import ModelCallback
from src.utils.logger import logging
from pathlib import Path
STAGE_NAME  = " Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):

        config = Configuration(Path("config\config.yaml"),Path("params.yaml"))
        get_config_data = config.get_training_config()
        model_training= ModelTraining(get_config_data)
        model_callbacks=ModelCallback()
        callbacks_list=model_callbacks._get_callbacks()
        model_training.get_base_model()
        model_training.train(callbacks_list)
        
if __name__ == "__main__":
    
    try:
        logging.info(f" >>>> stage {STAGE_NAME} <<<< started !")
        obj = ModelTrainingPipeline()
        obj.main()
        logging.info(f" >>>> stage {STAGE_NAME} <<<< Completed ! \n\n x==================x")
        
    except Exception as e:
        logging.exception(e)
        raise e
    