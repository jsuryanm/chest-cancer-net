from src.cancer_clf.config.configuration import ConfigurationManager
from src.cancer_clf.components.evaluation import Evaluation
from src.cancer_clf.logger.logger import logger 

STAGE_NAME = "Model Evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            eval_config = config.get_evaluation_config()
            eval = Evaluation(config=eval_config)
            eval.evaluation()
        
        except Exception as e:
            logger.exception(e)
            raise e
        
if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
