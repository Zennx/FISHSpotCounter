import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump
import matplotlib.pyplot as plt

# Paths
features_csv = None  # Precomputed features
results_csv = None  # Must map each image to the optimal K value
model_name = None

def xgbost_train(features_df, results_df, model_name):
    # Merge on Image column
    merged = features_df.merge(results_df, on="Image")
    feature_cols = [col for col in merged.columns if col.startswith("F")]
    X = merged[feature_cols].values
    y = merged["Optimum_K"].values
    w = merged["Weights"].values

    # QC: Plot histogram of K values (non-log)
    plt.figure()
    plt.hist(y, bins=30, color='orange', edgecolor='black')
    plt.xlabel('Optimum K Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Optimum K Values (Training Set)')
    plt.tight_layout()
    plt.savefig(model_name + "_K_histogram.png")
    plt.close()
    print(f"K value histogram saved as {model_name + '_K_histogram.png'}")
    print("K value variance:", np.var(y))
    print("K value mode:", pd.Series(y).mode().values)

    # Log-transform for training
    y_log = np.log1p(y)
    print("Log K value stats: min", np.min(y_log), "max", np.max(y_log), "mean", np.mean(y_log))

    print(f"Final Features: {len(X)}")
    print(f"Final Targets: {len(y)}")
    print(f"Final Weights: {len(w)}")

    # Split and train
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y_log, w, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_test)
    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.075, subsample=0.8, colsample_bytree=1.0, random_state=42)
    print("Training XGBoost model...")
    model.fit(X_train_scaled, y_train, sample_weight=w_train)

    # Evaluate
    y_pred = model.predict(X_val_scaled)  # Use scaled validation set
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model RMSE: {rmse:.4f}")
    # importances = model.feature_importances_
    # print("Feature importances:", importances)
    # Plot feature importances
    # plt.figure(figsize=(10, 6))
    # plt.bar(range(len(importances)), importances, color='orange', edgecolor='black')
    # plt.xlabel('Feature Index')
    # plt.ylabel('Importance')
    # plt.title('Feature Importances (XGBoost)')
    # plt.tight_layout()
    # plt.savefig(model_name + "_feature_importances.png")
    # plt.close()
    # print(f"Feature importances plot saved as {model_name + '_feature_importances.png'}")

    # Save model
    dump(model, model_name)
    dump(scaler, model_name + "_scaler.pkl")
    print("Trained model saved as '" + model_name +"'.")