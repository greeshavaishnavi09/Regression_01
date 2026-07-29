from Regression_01.config.configuration import ConfigurationManager
from Regression_01.components.data_validation import DataValidation

STAGE_NAME="DATA VALIDATION STAGE"

class DataValidationTrainingPipeline:

    def main(self):

        config = ConfigurationManager()

        validation_config = config.get_data_validation_config()

        validation = DataValidation(validation_config)

        validation.validate_dataset()