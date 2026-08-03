from Regression_01.config.configuration import ConfigurationManager
from Regression_01.components.model_evaluation import ModelEvaluation
from Regression_01.logging import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        model_evaluation_config = config.get_model_evaluation_config()

        model_evaluation = ModelEvaluation(
            config=model_evaluation_config
        )

        model_evaluation.initiate_model_evaluation()