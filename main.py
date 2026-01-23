from src.cancer_clf.logger.logger import logger
from src.cancer_clf.pipelines.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()

except Exception as e:
    logger.exception(e)
    raise e