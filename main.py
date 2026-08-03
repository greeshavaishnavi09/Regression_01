from Regression_01.pipeline.stage_01_dataingestion import DataIngestionTrainingPipeline
from Regression_01.pipeline.stage_02_datavalidation import DataValidationTrainingPipeline
from Regression_01.pipeline.stage_03_datatransformation import DataTransformationTrainingPipeline
from Regression_01.pipeline.stage_04_modeltrianer import ModelTrainerTrainingPipeline
from Regression_01.pipeline.stage_05_modelevaluation import ModelEvaluationTrainingPipeline


from Regression_01.logging import logger


STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<")

    obj = DataIngestionTrainingPipeline()
    obj.main()

    logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<")

    obj = DataValidationTrainingPipeline()
    obj.main()

    logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<")

    obj = DataTransformationTrainingPipeline()
    obj.main()

    logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Trainer Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<")

    obj = ModelTrainerTrainingPipeline()
    obj.main()

    logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Evaluation Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<")

    obj = ModelEvaluationTrainingPipeline()
    obj.main()

    logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<")

except Exception as e:
    logger.exception(e)
    raise e