from src.config.configurtion import Configuration
from src.components.model_base import BaseModel
from src.utils.logger import logging
from pathlib import Path
STAGE_NAME  = "Model Base Stage"

class BaseModelTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):

        config = Configuration(Path("config\config.yaml"),Path("params.yaml"))
        get_config_data = config.get_base_model_config()
        model_base = BaseModel(get_config_data)
        model_base.get_base_model()
        model_base.update_base_model()

if __name__ == "__main__":
    
    try:
        logging.info(f" >>>> stage {STAGE_NAME} <<<< started !")
        obj = BaseModelTrainingPipeline()
        obj.main()
        logging.info(f" >>>> stage {STAGE_NAME} <<<< Completed ! \n\n x==================x")
        
    except Exception as e:
        logging.exception(e)
        raise e
    