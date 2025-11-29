"""
Housing Price Prediction
Models Used: Linear Regression and XGBoost (XGRegressor)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

class PricePredictor:
    """
    Prepares data, provides linear regression & XGRegressor models, and exports data
    """
    def __init__(self, data_path):
        """
        Sets up file paths and initializes models
        
        :param data_path: Dataset path
        """
        self.data_path = data_path

        # initialising the models
        self.lr_model = LinearRegression()
        self.xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1, random_state=1)
                
        # holds data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """
        Loads data, cleans it (get's required categories), and splits into train/test sets
        """
        df = pd.read_csv(self.data_path)

        # cleans column names for ease
        df.columns = df.columns.str.replace(' ', '')

        features = ['LotArea', 'YearBuilt', 'OverallQual', 'Neighborhood']
        target = 'SalePrice'

        # creates binary matrix for categorical data
        df_processed = pd.get_dummies(df[features + [target]], columns=['Neighborhood'], drop_first=True)

        # split target from data
        X = df_processed.drop(target, axis=1)
        y = df_processed[target]

        # 80% train 20% test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    def linear_regression(self):
        """
        Trains linear model and gets top influential features
        """
        self.lr_model.fit(self.X_train, self.y_train)

        # holds coefficients
        coef_df = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Coefficient': self.lr_model.coef_
        })

        # sorts by impact to see important features
        coef_df['Abs_Coeff'] = coef_df['Coefficient'].abs()
        print("Top 3 Important Features (Linear Model):")
        print(coef_df.sort_values(by='Abs_Coeff', ascending=False).head(3))

        # evaluation
        y_pred = self.lr_model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        return rmse

    def xgb_regressor(self):
        """
        Trains XGBRegressor model and returns results
        """
        self.xgb_model.fit(self.X_train, self.y_train)

        # evaluation
        y_pred = self.xgb_model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        return rmse, y_pred

    def export_value_gap(self, predictions, filename='housing_value_gap.csv'):
        """
        Merges original data with predictions and calculates (predicted - actual) price
        
        :param predictions: Predicted prices
        :param filename: Output csv filename
        """
        # loads test data    
        raw_df = pd.read_csv(self.data_path)
        df_export = raw_df.loc[self.y_test.index].copy()
        df_export.columns = df_export.columns.str.replace(' ', '')
        
        features = ['LotArea', 'YearBuilt', 'OverallQual', 'Neighborhood', 'SalePrice']
        df_export = df_export[features]
        
        # adds predictions
        df_export['Predicted_Price'] = predictions

        # Positive = Undervalued
        # Negative = Overvalued
        df_export['Value_Gap'] = df_export['Predicted_Price'] - self.y_test

        # exports
        df_export.to_csv(filename, index=False)
        print(f"Data exported to: {filename}")