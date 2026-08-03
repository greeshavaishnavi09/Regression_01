import os
import pandas as pd
from Regression_01.logging import logger
import numpy as np

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor
)

from xgboost import XGBRegressor

from Regression_01.entity.config_entity import ModelTrainerConfig

from Regression_01.utils.common import (
    evaluate_models,
    save_object
)


class ModelTrainer:

    def __init__(
        self,
        config: ModelTrainerConfig
    ):

        self.config = config


    def initiate_model_trainer(self):

        # STEP 1 : Load transformed train & test datasets

        logger.info("Loading transformed train and test arrays")

        train_arr = np.load(
            self.config.transformed_train_path
        )

        test_arr = np.load(
            self.config.transformed_test_path
        )


        # STEP 2 : Split Features and Target

        logger.info("Splitting train and test data")

        X_train = train_arr[:, :-1]
        y_train = train_arr[:, -1]

        X_test = test_arr[:, :-1]
        y_test = test_arr[:, -1]

        # STEP 3 : Create Regression Models

        logger.info("Creating regression models")

        models = {

            "Linear Regression": LinearRegression(),

            "Ridge Regression": Ridge(),

            "Lasso Regression": Lasso(),

            "Decision Tree": DecisionTreeRegressor(
                random_state=42
            ),

            "Random Forest": RandomForestRegressor(
                random_state=42
            ),

            "XGBoost": XGBRegressor(
                random_state=42,
                verbosity=0
            ),

            "AdaBoost": AdaBoostRegressor(
                random_state=42
            )

        }

        # STEP 4 : Hyperparameter Dictionary


        logger.info("Creating Hyperparameter Dictionary")

        params = {

            "Linear Regression": {},

            "Ridge Regression": {

                "alpha":[0.01,0.1,1,10]

            },

            "Lasso Regression": {

                "alpha":[0.001,0.01,0.1,1]

            },

            "Decision Tree": {

                "max_depth":[3,5,10],

                "min_samples_split":[2,5]

            },

            "Random Forest": {

                "n_estimators":[100,200],

                "max_depth":[5,10]

            },

            "XGBoost": {

                "n_estimators":[100,200],

                "learning_rate":[0.01,0.1],

                "max_depth":[3,5]

            },

            "AdaBoost": {

                "n_estimators":[50,100,200],

                "learning_rate":[0.01,0.1,1]

            }

        }


        # STEP 5 : Train Models + GridSearchCV


        logger.info("Training all regression models")

        report, trained_models, comparison_df = evaluate_models(

            X_train=X_train,

            y_train=y_train,

            X_test=X_test,

            y_test=y_test,

            models=models,

            params=params

        )


        # STEP 6 : Display Model Comparison

        logger.info("Model Comparison")

        print(comparison_df)

        # STEP 7 : Save Comparison Table
        
        logger.info("Saving Model Comparison Report")
        
        comparison_df.to_csv(
        
        self.config.root_dir / "model_report.csv",
        
        index=False
        )

        # STEP 8 : Select Best Model#

        logger.info("Selecting best model")

        best_model_name = comparison_df.iloc[0]["Model"]

        best_model = trained_models[best_model_name]

        best_model_score = comparison_df.iloc[0]["R2 Score"]

        logger.info(
            f"Best Model : {best_model_name}"
        )

        logger.info(
            f"Best R2 Score : {best_model_score:.4f}"
        )


        # STEP 9 : Save Best Model

        logger.info("Saving best model")

        save_object(

            file_path=self.config.trained_model_file_path,

            obj=best_model

        )

        logger.info("Best model saved successfully")


        # STEP 10 : Return Comparison Table

        return comparison_df
