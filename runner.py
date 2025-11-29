"""
Runner for the Prediction program
Dataset used for sample run: Ames Housing Dataset
Access: https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset/
"""

from price_predictor import PricePredictor

def main(data_path):
    print("Starting Housing Analysis...")
    predictor = PricePredictor(data_path)

    # load & prepare data
    predictor.load_data()

    # linear reg. model
    lr_rmse = predictor.linear_regression()
    print(f"Linear Regression RMSE: ${lr_rmse:,.3f}")

    # xgbregressor model
    xgb_rmse, predictions = predictor.xgb_regressor()
    print(f"XGBoost RMSE: ${xgb_rmse:,.3f}")

    # compare lin. reg. vs xgb
    improvement = lr_rmse - xgb_rmse
    print(f"Model Improvement: ${improvement:,.3f}")

    # export data
    predictor.export_value_gap(predictions)
    print("Process Complete.")

# runner
data_path = input("Enter the path of the data (.csv) here: ")
main(data_path)