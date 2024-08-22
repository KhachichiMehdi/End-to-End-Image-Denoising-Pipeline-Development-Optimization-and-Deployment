from src.config.configurtion import Configuration
from src.components.data_preprocessing import DataPreprocessing
from src.utils.logger import logging
from pathlib import Path
STAGE_NAME  = "Data preprocessing Stage"

class DataPreprocessingTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):

        config = Configuration(Path("config\config.yaml"),Path("params.yaml"))
        get_config_data = config.get_data_preprocessing_config()
        data_preprocessing = DataPreprocessing(get_config_data)
        
if __name__ == "__main__":
    
    try:
        logging.info(f" >>>> stage {STAGE_NAME} <<<< started !")
        obj = DataPreprocessingTrainingPipeline()
        obj.main()
        logging.info(f" >>>> stage {STAGE_NAME} <<<< Completed ! \n\n x==================x")
        
    except Exception as e:
        logging.exception(e)
        raise e
    