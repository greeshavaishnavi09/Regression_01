import pandas as pd
import joblib
from Regression_01.logging import logger
from Regression_01.entity.config_entity import PredictionPipelineConfig

# components


class PredictionPipeline:

    def __init__(self, config: PredictionPipelineConfig):
        self.config = config

    def predict(self, features):

        logger.info("Loading preprocessor")
        preprocessor = joblib.load(self.config.preprocessor_path)

        logger.info("Loading trained model")
        model = joblib.load(self.config.model_path)

        logger.info("Creating input DataFrame")
        input_df = pd.DataFrame([features])

        logger.info("Transforming input data")
        transformed_input = preprocessor.transform(input_df)

        logger.info("Making prediction")
        prediction = model.predict(transformed_input)

        logger.info(f"Prediction: {prediction[0]}")

        return prediction[0]