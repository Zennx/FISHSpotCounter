import numpy as np
from skimage import exposure, filters, measure
import cv2
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from core.spot_counter import detect_spots_log, save_spot_overlay

model = None  # Placeholder for the model, should be loaded externally
model = joblib.load("C:/Users/ONG32/trained_model_XGB_V3.1.pkl")

def extract_features(image):
    # Extract basic intensity features from an image
    mean = np.mean(image)
    std = np.std(image)
    max_val = np.max(image)
    p75 = np.percentile(image, 75)
    nonzero_count = np.count_nonzero(image)
    nonzero_ratio = nonzero_count / image.size
    entropy = measure.shannon_entropy(image)
    otsu_thresh = filters.threshold_otsu(image)
    return [mean, std, max_val, p75, nonzero_count, nonzero_ratio, entropy, otsu_thresh]

def rescaled (image, features, k):
    return exposure.rescale_intensity(image, in_range=(features[0]*k, 225))

def apply_otsu_threshold(image):
    """Apply Otsu thresholding."""
    threshold_val = filters.threshold_otsu(image)
    return image > threshold_val

def process_image(file_path, output_dir, save_overlay=True):
    filename = os.path.basename(file_path)
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"Failed to read {file_path}, skipping.")
        return None

    features = extract_features(img)
    k = model.predict([features])[0]
    flattened = rescaled(img, features, k)
    entropy = measure.shannon_entropy(flattened)

    if entropy > 0.03 and k > 2.5:
        smoothed = cv2.GaussianBlur(flattened, (3, 3), 0)
        filtered = apply_otsu_threshold(smoothed)
    else:
        filtered = flattened

    # This still applies some Gaussian smoothing but weaker
    filtered = cv2.GaussianBlur(flattened, (3, 3), sigmaX=0.65)

    blobs = detect_spots_log(filtered)

    if save_overlay:
        overlay_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_overlay.png")
        save_spot_overlay(flattened, blobs, overlay_path, title=f"Detected Spots - {filename}")

    return {
        "Image": filename,
        "SpotCount": len(blobs),
        "KValue": k,
        "Entropy": entropy
    }

def process_images_batch(file_paths, output_dir, save_overlay=True):
    """
    Process a batch of images: extract features, predict k in batch, and process each image.
    Returns a list of result dicts.
    """
    images = []
    filenames = []
    features_list = []

    # Load images and extract features
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Failed to read {file_path}, skipping.")
            images.append(None)
            filenames.append(filename)
            features_list.append(None)
            continue
        images.append(img)
        filenames.append(filename)
        features_list.append(extract_features(img))

    # Prepare feature matrix for valid images
    valid_indices = [i for i, f in enumerate(features_list) if f is not None]
    feature_matrix = np.array([features_list[i] for i in valid_indices])

    # Predict k values in batch
    k_values = np.zeros(len(images))
    if len(feature_matrix) > 0:
        k_pred = model.predict(feature_matrix)
        for idx, k in zip(valid_indices, k_pred):
            k_values[idx] = k

    results = []
    for i, img in enumerate(images):
        filename = filenames[i]
        features = features_list[i]
        if img is None or features is None:
            continue
        k = k_values[i]
        flattened = rescaled(img, features, k)
        entropy = measure.shannon_entropy(flattened)

        if entropy > 0.03 and k > 2.5:
            smoothed = cv2.GaussianBlur(flattened, (3, 3), 0)
            filtered = apply_otsu_threshold(smoothed)
        else:
            filtered = flattened

        filtered = cv2.GaussianBlur(flattened, (3, 3), sigmaX=0.65)
        blobs = detect_spots_log(filtered)

        if save_overlay:
            overlay_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_overlay.png")
            save_spot_overlay(flattened, blobs, overlay_path, title=f"Detected Spots - {filename}")

        results.append({
            "Image": filename,
            "SpotCount": len(blobs),
            "KValue": k,
            "Entropy": entropy
        })
    return results