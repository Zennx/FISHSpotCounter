import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump

# Paths
features_csv = "F:/Kromatid/TrainingKOptimiserV3Test/features.csv"  # Precomputed features
results_csv = "F:/Kromatid/TrainingKOptimiserV3Test/k_results.csv"  # Must map each image to the optimal K value
model_name = "trained_model_XGB_V3.1.pkl"

def xgbost_train(features_df, results_df, model_name):
    # Merge on Image column
    merged = features_df.merge(results_df, on="Image")
    feature_cols = [col for col in merged.columns if col.startswith("F")]
    X = merged[feature_cols].values
    y = merged["Optimum_K"].values
    w = merged["Weights"].values

    print(f"Final Features: {len(X)}")
    print(f"Final Targets: {len(y)}")
    print(f"Final Weights: {len(w)}")

    # Split and train
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=80, max_depth=5, learning_rate=0.05)
    model.fit(X_train, y_train, sample_weight=w_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model RMSE: {rmse:.4f}")

    # Save model
    dump(model, model_name)
    print("Trained model saved as '" + model_name +"'.")