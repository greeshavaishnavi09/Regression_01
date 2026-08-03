import pandas as pd

from Regression_01.logging import logger
from Regression_01.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:

    def __init__(
        self,
        config: ModelEvaluationConfig
    ):

        self.config = config


    def initiate_model_evaluation(self):

        # STEP 1 : Load Model Comparison Report

        logger.info("Loading model comparison report")

        comparison_df = pd.read_csv(
            self.config.model_report_path
        )

        # STEP 2 : Get Best Model

        logger.info("Finding best model")

        best_model = comparison_df.iloc[0]

        best_model_name = best_model["Model"]

        best_r2_score = best_model["R2 Score"]


        # STEP 3 : Read Threshold

        logger.info("Reading evaluation threshold")

        threshold = self.config.threshold


        # STEP 4 : Compare Best Model with Threshold

        logger.info("Evaluating model")

        if best_r2_score >= threshold:

            status = "Accepted"

        else:

            status = "Rejected"


        # STEP 5 : Create Evaluation Report

        logger.info("Creating evaluation report")

        evaluation_df = pd.DataFrame({

            "Best Model":[best_model_name],

            "R2 Score":[best_r2_score],

            "Threshold":[threshold],

            "Status":[status]

        })

        # STEP 6 : Save Evaluation Report

        if self.config.save_metrics:

            logger.info("Saving evaluation report")

            evaluation_df.to_csv(

                self.config.evaluation_report_path,

                index=False

            )

            logger.info("Evaluation report saved successfully")

        # STEP 7 : Print Final Status

        logger.info(f"Best Model : {best_model_name}")

        logger.info(f"R2 Score : {best_r2_score:.4f}")

        logger.info(f"Status : {status}")


        return evaluation_df