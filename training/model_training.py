import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from joblib import dump
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, normaltest

# Paths
features_csv = None  # Precomputed features
results_csv = None  # Must map each image to the optimal K value
model_name = None

def bin_k_values(k_values, bin_width=0.05, k_min=0.5, k_max=3.0):
    bins = np.arange(k_min, k_max + bin_width, bin_width)
    bin_indices = np.digitize(k_values, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    return bin_indices, bins

def make_soft_labels(working_ks_list, bins):
    n_samples = len(working_ks_list)
    n_bins = len(bins) - 1
    soft_labels = np.zeros((n_samples, n_bins), dtype=np.float32)
    for i, ks in enumerate(working_ks_list):
        if not ks or (isinstance(ks, float) and np.isnan(ks)):
            continue
        bin_indices = np.digitize(ks, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        for idx in np.unique(bin_indices):
            soft_labels[i, idx] = 1.0
        if np.sum(soft_labels[i]) > 0:
            soft_labels[i] /= np.sum(soft_labels[i])
    return soft_labels

def xgbost_train(features_df, results_df, model_name):
    # Merge on Image column
    merged = features_df.merge(results_df, on="Image")
    feature_cols = [col for col in merged.columns if col.startswith("F")]
    X = merged[feature_cols].values
    working_ks_list = merged["Working_Ks"].apply(lambda x: eval(x) if isinstance(x, str) else x).tolist()
    y_raw = merged["Median_K"].values

    bin_width = 0.05
    k_min = 0.5
    k_max = 3.0
    _, bins = bin_k_values([k_min, k_max], bin_width, k_min, k_max)
    n_bins = len(bins) - 1

    y_soft = make_soft_labels(working_ks_list, bins)

    # QC: Plot histogram of K values (non-log)
    k_kurt = kurtosis(y)
    k_skew = skew(y)
    k_norm_stat, k_norm_p = normaltest(y)

    plt.figure()
    plt.hist(y_raw, bins=30, color='orange', edgecolor='black')
    plt.xlabel('Median K Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Optimum K Values (Training Set)')
    # Annotate stats in top left
    stats_text = f"Kurtosis: {k_kurt:.2f}\nSkewness: {k_skew:.2f}\nNormality p: {k_norm_p:.3g}"
    plt.gca().text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.tight_layout()
    plt.savefig(model_name + "_median_K_histogram.png")
    plt.close()
    print(f"K value histogram saved as {model_name + '_K_histogram.png'}")
    print("K value variance:", np.var(y))
    print("K value mode:", pd.Series(y).mode().values)
    print(f"Kurtosis: {k_kurt:.3f}")
    print(f"Skewness: {k_skew:.3f}")
    print(f"Normality test p-value: {k_norm_p:.3g}")

    # Log-transform for training
    y_log = np.log1p(y)
    print("Log K value stats: min", np.min(y_log), "max", np.max(y_log), "mean", np.mean(y_log))

    print(f"Final Features: {len(X)}")
    print(f"Final Classes (bins): {n_bins}")

    # Split and train
    X_train, X_test, y_train_soft, y_test_soft = train_test_split(X, y_soft, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_test)

    y_train_hard = np.argmax(y_train_soft, axis=1)
    y_test_hard = np.argmax(y_test_soft, axis=1)

    # Ensure all classes are present in y_train_hard
    all_classes = np.arange(n_bins)
    missing_classes = set(all_classes) - set(np.unique(y_train_hard))
    if missing_classes:
        print(f"Warning: Missing classes in training set: {sorted(missing_classes)}. Adding one sample for each missing class.")
        mean_feat = np.mean(X_train_scaled, axis=0)
        X_train_scaled = np.vstack([X_train_scaled] + [mean_feat for _ in missing_classes])
        y_train_hard = np.concatenate([y_train_hard, list(missing_classes)])

    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.075,
        subsample=0.8, colsample_bytree=1.0, random_state=42,
        objective="multi:softprob", num_class=n_bins
    )
    print("Training XGBoost classifier (with hard label argmax)...")
    model.fit(X_train_scaled, y_train_hard)

    # Evaluate
    y_pred_proba = model.predict_proba(X_val_scaled)
    y_pred = np.argmax(y_pred_proba, axis=1)
    acc = accuracy_score(y_test_hard, y_pred)
    # Explicitly provide all possible labels to log_loss to avoid ValueError
    labels = np.arange(n_bins)
    lloss = log_loss(y_test_hard, y_pred_proba, labels=labels)
    print(f"Model accuracy: {acc:.4f}")
    print(f"Model log loss (hard labels): {lloss:.4f}")

    # Save model, scaler, and bins
    dump(model, model_name)
    dump(scaler, model_name + "_scaler.pkl")
    print("Trained classifier saved as '" + model_name +"'.")