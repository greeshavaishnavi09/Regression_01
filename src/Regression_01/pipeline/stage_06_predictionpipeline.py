from Regression_01.config.configuration import ConfigurationManager
from Regression_01.components.prediction_pipeline import PredictionPipeline
from Regression_01.logging import logger

# pipeline

class PredictionPipelineTrainingPipeline:

    def __init__(self):
        pass

    def main(self, input_data):

        try:
            logger.info(">>>>> Prediction Pipeline Stage Started <<<<<")

            config = ConfigurationManager()

            prediction_pipeline_config = (
                config.get_prediction_pipeline_config()
            )

            prediction_pipeline = PredictionPipeline(
                config=prediction_pipeline_config
            )

            prediction = prediction_pipeline.predict(input_data)

            logger.info(">>>>> Prediction Pipeline Stage Completed <<<<<")

            return prediction

        except Exception as e:
            logger.exception(e)
            raise e