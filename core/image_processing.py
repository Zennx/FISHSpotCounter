import numpy as np
from skimage import exposure, filters, measure, feature
from scipy.stats import skew, kurtosis
import cv2
import os
import joblib
from core.spot_counter import detect_spots_log, detect_spots_doh, save_spot_overlay
from concurrent.futures import ThreadPoolExecutor
from core.file_handler import get_mask_path

parallelisation_enabled = True  # Set to True/False to enable/disable parallel feature extraction
laplacian_sharpening_enabled = True  # Set to True/False to enable/disable Laplacian sharpening

def extract_features(image): #all features for feature selection
    # Extract basic intensity features from an image
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    image = clahe.apply(image.astype(np.uint8))
    image = np.clip(image + 0, 0, 255)
    mean = np.mean(image)
    std = np.std(image)
    max_val = np.max(image)
    p75 = np.percentile(image, 75)
    nonzero_count = np.count_nonzero(image)
    entropy = measure.shannon_entropy(image)
    otsu_thresh = filters.threshold_otsu(image)
    ppxmean = (image > image.mean()).sum() / image.size
    ppx74 = (image > np.percentile(image, 74)).sum() / image.size
    snr = (image.max() - image.mean()) / image.std() if std > 0 else 0
    contrast = cv2.Laplacian(image, cv2.CV_64F).var()  # Variance of Laplacian for contrast
    skewness = skew(image.reshape(-1))  # Skewness of the image
    kurt = kurtosis(image.reshape(-1))  # Kurtosis of the image
    img_unit8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edge_density = np.sum(cv2.Canny(img_unit8, 50, 150)) / image.size  # Edge density
    coarseness = np.mean(np.abs(np.diff(image, axis=0))) + np.mean(np.abs(np.diff(image, axis=1)))  # Coarseness
    lbp = feature.local_binary_pattern(image, P=8, R=1, method='uniform')  # Local Binary Pattern
    lbp_uniform = np.sum(lbp <= 8) / image.size
    return [mean, std, max_val, p75, nonzero_count, entropy, otsu_thresh,
            ppxmean, ppx74, snr, contrast, skewness, kurt, edge_density, coarseness, lbp_uniform]

def extract_features_all(image): #all features for feature selection
    # Extract basic intensity features from an image
    mean = np.mean(image)
    std = np.std(image)
    max_val = np.max(image)
    p75 = np.percentile(image, 75)
    nonzero_count = np.count_nonzero(image)
    nonzero_ratio = nonzero_count / image.size
    entropy = measure.shannon_entropy(image)
    otsu_thresh = filters.threshold_otsu(image)
    ppxmean = (image > image.mean()).sum() / image.size
    ppx74 = (image > np.percentile(image, 74)).sum() / image.size
    snr = (image.max() - image.mean()) / image.std() if std > 0 else 0
    contrast = cv2.Laplacian(image, cv2.CV_64F).var()  # Variance of Laplacian for contrast
    skewness = skew(image.reshape(-1))  # Skewness of the image
    kurt = kurtosis(image.reshape(-1))  # Kurtosis of the image
    img_unit8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edge_density = np.sum(cv2.Canny(img_unit8, 50, 150)) / image.size  # Edge density
    coarseness = np.mean(np.abs(np.diff(image, axis=0))) + np.mean(np.abs(np.diff(image, axis=1)))  # Coarseness
    lbp = feature.local_binary_pattern(image, P=8, R=1, method='uniform')  # Local Binary Pattern
    lbp_uniform = np.sum(lbp <= 8) / image.size
    edge_density_lbp = np.sum(lbp > 8) / image.size  # Edge density of LBP
    return [mean, std, max_val, p75, nonzero_count, nonzero_ratio, entropy, otsu_thresh,
            ppxmean, ppx74, snr, contrast, skewness, kurt, edge_density, coarseness, lbp_uniform, edge_density_lbp]


def rescaled_Oligo(image, features, k):
    return exposure.rescale_intensity(image, in_range=(features[0]*k, 225))

def rescaled_BAC(image, features, k):
    return exposure.rescale_intensity(image, in_range=(features[0]*k, 255))

def apply_otsu_threshold(image):
    """Apply Otsu thresholding."""
    threshold_val = filters.threshold_otsu(image)
    return image > threshold_val

def laplacian_sharpening(image):     # Apply mild Laplacian sharpening, 0 for none
    # Ensure image is float32 or uint8 for OpenCV compatibility
    if laplacian_sharpening_enabled:
        img = image.astype(np.float32) if image.dtype not in [np.uint8, np.float32] else image
        laplacian_sharpened = cv2.Laplacian(img, cv2.CV_32F)
        sharpened = cv2.convertScaleAbs(img + (1e-7 * laplacian_sharpened))
        normalised = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
    else:
        normalised = image
    return normalised

def process_images_batch(
    file_paths, output_dir, save_overlay=True, model_path=None, scaler_path=None,
    parallel_features=parallelisation_enabled, masks_dir=None, probe_type="BAC"
):
    """
    Process a batch of images: extract features, predict k in batch, and process each image.
    Returns a list of result dicts.
    """
    if not model_path or not os.path.isfile(model_path):
        raise ValueError("A valid model_path (.pkl) must be provided.")
    model = joblib.load(model_path)
    scaler = None
    if scaler_path and os.path.isfile(scaler_path):
        scaler = joblib.load(scaler_path)

    images = []
    filenames = []
    masks = []

    # Load images
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        images.append(img)
        filenames.append(filename)
        # Load mask if masks_dir is provided
        if masks_dir:
            mask_path = get_mask_path(file_path, masks_dir)
            if os.path.isfile(mask_path):
                mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                masks.append(mask_img)
            else:
                masks.append(None)
        else:
            masks.append(None)

    # Feature extraction (optionally parallel)
    def safe_extract(img):
        if img is None:
            return None
        try:
            return extract_features(img)
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            return None

    if parallel_features:
        with ThreadPoolExecutor() as executor:
            features_list = list(executor.map(safe_extract, images))
    else:
        features_list = [safe_extract(img) for img in images]

    # Prepare feature matrix for valid images
    valid_indices = [i for i, f in enumerate(features_list) if f is not None]
    feature_matrix = np.array([features_list[i] for i in valid_indices])

    # Apply scaler if provided
    if scaler is not None and len(feature_matrix) > 0:
        feature_matrix = scaler.transform(feature_matrix)
    else:
        if len(feature_matrix) > 0:
            print("No scaler provided or no valid features to scale.")

    # Predict k values in batch
    k_values = np.zeros(len(images))
    if len(feature_matrix) > 0:
        try:
            k_pred_log = model.predict(feature_matrix)
            k_pred = np.expm1(k_pred_log)  # Convert log(K) back to K
            for idx, k in zip(valid_indices, k_pred):
                k_values[idx] = k
        except Exception as e:
            print(f"Model prediction failed: {e}")

    results = []
    for i, img in enumerate(images):
        filename = filenames[i]
        features = features_list[i]
        if img is None or features is None:
            continue
        k = k_values[i]

        # --- Defensive: Always set flattened, and handle unknown probe_type ---
        probe_type_str = str(probe_type).strip().upper()
        if probe_type_str == "BAC":
            flattened = rescaled_BAC(img, features, k)
        elif probe_type_str == "OLIGO":
            flattened = rescaled_Oligo(img, features, k)
        else:
            print(f"Warning: Unknown probe_type '{probe_type}', defaulting to BAC rescaling.")
            flattened = rescaled_BAC(img, features, k)

        entropy = measure.shannon_entropy(flattened)

        if entropy > 0.03 and k > 2.5:
            smoothed = cv2.GaussianBlur(flattened.astype(np.uint8), (3, 3), 0)
            filtered = apply_otsu_threshold(smoothed)
            # Ensure filtered is float32 for further OpenCV ops
            filtered = filtered.astype(np.float32)
        else:
            filtered = flattened.astype(np.float32)

        filtered = cv2.GaussianBlur(filtered, (3, 3), sigmaX=0.65)
        sharpened = laplacian_sharpening(filtered)

        if probe_type_str == "OLIGO":
            blobs = detect_spots_log(sharpened)
        elif probe_type_str == "BAC":
            blobs = detect_spots_doh(sharpened)
        else:
            blobs = detect_spots_log(sharpened)

        # --- Masking logic ---
        mask_img = masks[i]
        if mask_img is not None and blobs is not None and len(blobs) > 0:
            # Assume mask is binary (nonzero = inside mask)
            filtered_blobs = []
            for blob in blobs:
                y, x, r = blob
                y_int, x_int = int(round(y)), int(round(x))
                if 0 <= y_int < mask_img.shape[0] and 0 <= x_int < mask_img.shape[1]:
                    if mask_img[y_int, x_int] > 0:
                        filtered_blobs.append(blob)
            blobs = np.array(filtered_blobs)

        if save_overlay:
            overlay_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_overlay.png")
            save_spot_overlay(flattened, blobs, overlay_path, title=f"Detected Spots - {filename}")

        results.append({
            "Image": filename,
            "SpotCount": len(blobs) if blobs is not None else 0,
            "KValue": k,
            "Entropy": entropy
        })
    return results