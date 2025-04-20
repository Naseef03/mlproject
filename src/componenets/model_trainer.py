import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1]
            models = {
                "Random Forest Regressor": RandomForestRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "XGBoost Regressor": XGBRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "KNN Regressor": KNeighborsRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression()
            }

            params = {
                "Decision Tree Regressor" : {
                    "criterion": ["squared_error", "absolute_error", "poisson", "friedman_mse"]
                },
                "Random Forest Regressor": {
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting Regressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "KNN Regressor": {
                    "n_neighbors": [5, 7, 9, 11]
                },
                "XGBoost Regressor": {
                    "learning_rate": [.1, .01, .05, .001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "CatBoost Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [.1, .01, .05],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                }
            }

            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models, params)

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = max(model_report.values())
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2 = r2_score(y_test, predicted)
    
            return r2

        except Exception as e:
            raise CustomException(e, sys)
            best_model_score = max(model_report.values())
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2 = r2_score(y_test, predicted)
        
            return r2

        except Exception as e:
            raise CustomException(e, sys)


