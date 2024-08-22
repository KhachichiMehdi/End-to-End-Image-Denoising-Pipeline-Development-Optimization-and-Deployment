from src.config.configurtion import Configuration
from src.components.data_ingestion import DataIngestion
from src.utils.logger import logging
from pathlib import Path
STAGE_NAME  = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):

        config = Configuration(Path("config\config.yaml"),Path("params.yaml"))
        get_config_data = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(get_config_data)
        data_ingestion.initiate_data_ingestion()
        
if __name__ == "__main__":
    
    try:
        logging.info(f" >>>> stage {STAGE_NAME} <<<< started !")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logging.info(f" >>>> stage {STAGE_NAME} <<<< Completed ! \n\n x==================x")
        
    except Exception as e:
        logging.exception(e)
        raise e
    