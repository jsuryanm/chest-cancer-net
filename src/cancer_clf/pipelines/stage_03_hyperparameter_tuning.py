from src.cancer_clf.config.configuration import ConfigurationManager
from src.cancer_clf.components.hyperparameter_tuning import HyperparameterTuning
from src.cancer_clf.logger.logger import logger

STAGE_NAME = "Hyperparameter Tuning Stage"

class HyperparameterTuningTrainingPipeline:
    def __init__(self):
        pass 

    def main(self):
        config = ConfigurationManager()
        
        tuning_config = config.get_hyperparameter_tuning_config()
        
        tuner = HyperparameterTuning(config=tuning_config)
        tuner.run()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = HyperparameterTuningTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n")
    
    except Exception as e:
        logger.exception(e)
        raise e 
