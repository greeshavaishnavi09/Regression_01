from pathlib import Path
import yaml
import sys
import os
import json
import joblib
from box import ConfigBox
from box.exceptions import BoxValueError
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from ensure import ensure_annotations

from Regression_01.logging import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads YAML file and returns ConfigBox object.

    Args:
        path_to_yaml (Path): Path to YAML file

    Returns:
        ConfigBox: Parsed YAML content
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose = True):
    """create list of directories
    
    args:
        path to directories (list) : list of path to directories
        ignore_log (bool, optional): ignore if multiple directories is to be created. default to false
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at {path}")


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in kb

    args:
        path (Path): path of the file
        
    Return:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


@ensure_annotations
def save_json(path: Path, data: dict):
    """saves a dictionary as a json file

    Args:
        path (Path): path to the destination json file
        data (dict): dictionary data to be saved
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved successfully at: {path}")


@ensure_annotations
def save_object(file_path, obj):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    joblib.dump(obj, file_path)

    logger.info(f"Object saved at {file_path}")


@ensure_annotations  
def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    params
):

    report = {}

    trained_models = {}

    comparison = []

    for model_name, model in models.items():

        logger.info(f"Training {model_name}")

        param_grid = params.get(model_name, {})

        # Hyperparameter Tuning
        if param_grid:

            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                scoring="r2",
                n_jobs=-1
            )

            grid.fit(X_train, y_train)

            model = grid.best_estimator_

        # Train Model
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)

        mse = mean_squared_error(y_test, y_pred)

        rmse = np.sqrt(mse)

        r2 = r2_score(y_test, y_pred)

        report[model_name] = r2

        trained_models[model_name] = model

        comparison.append({

            "Model": model_name,

            "MAE": mae,

            "RMSE": rmse,

            "R2 Score": r2

        })

    comparison_df = pd.DataFrame(comparison)

    comparison_df = comparison_df.sort_values(
        by="R2 Score",
        ascending=False
    )

    return report, trained_models, comparison_df  
    