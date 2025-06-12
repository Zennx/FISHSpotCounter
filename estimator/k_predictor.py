import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from skimage import measure
from glob import glob
import pandas as pd
from tqdm import tqdm

from core.image_processing import extract_features, rescaled, apply_otsu_threshold
from core.spot_counter import detect_spots_log

EXPECTED_SPOTS = 2
K_RANGE = (0.5, 3.0)
K_STEP = 0.05
K_FINE_STEP = 0.01


def count_spots_at_k(img, features, k):
    flattened = rescaled(img, features, k)
    entropy = measure.shannon_entropy(flattened)

    if entropy > 0.03 and k > 2.5:
        smoothed = cv2.GaussianBlur(flattened, (3, 3), 0)
        filtered = apply_otsu_threshold(smoothed)
    else:
        filtered = flattened

    filtered = filtered.astype(np.uint8) * 255
    filtered = cv2.GaussianBlur(filtered, (3, 3), sigmaX=0.65)
    blobs = detect_spots_log(filtered)
    return len(blobs)


def find_longest_contiguous_k(spot_counts):
    max_len = 0
    best_k = None

    current_len = 0
    current_start = None

    for k, count in spot_counts:
        if count == EXPECTED_SPOTS:
            if current_start is None:
                current_start = k
                current_len = 1
            else:
                current_len += 1
            if current_len > max_len:
                max_len = current_len
                best_k = current_start + ((current_len - 1) * K_STEP) / 2
        else:
            current_len = 0
            current_start = None

    if best_k is None:
        valid_k = [k for k, count in spot_counts if count >= EXPECTED_SPOTS]
        if valid_k:
            best_k = max(valid_k)

    return best_k


def find_optimal_k(img):
    features = extract_features(img)  # Extract once
    ks = np.arange(K_RANGE[0], K_RANGE[1] + K_STEP, K_STEP)
    spot_counts = [(k, count_spots_at_k(img, features, k)) for k in ks]

    best_k = find_longest_contiguous_k(spot_counts)

    if best_k is None:
        # Try finer resolution if no valid K found
        ks_fine = np.arange(K_RANGE[0], K_RANGE[1] + K_FINE_STEP, K_FINE_STEP)
        spot_counts = [(k, count_spots_at_k(img, features, k)) for k in ks_fine]
        best_k = find_longest_contiguous_k(spot_counts)

    return best_k, spot_counts


def plot_k_vs_spots(spot_counts, best_k, filename, output_dir):
    ks, counts = zip(*spot_counts)
    plt.figure()
    plt.plot(ks, counts, label="Spot count")
    plt.axhline(EXPECTED_SPOTS, color='gray', linestyle='--', label="Expected")
    if best_k is not None:
        plt.axvline(best_k, color='red', linestyle='--', label="Best K")
    plt.xlabel("K")
    plt.ylabel("Detected Spots")
    plt.title(f"Spot Count vs K - {filename}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}_Kplot.png"))
    plt.close()


def compute_weight(variance):
    return 1 / (1 + variance)


def run_k_predictor(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    input_files = glob(os.path.join(input_dir, "*.tif")) + \
                  glob(os.path.join(input_dir, "*.tiff")) + \
                  glob(os.path.join(input_dir, "*.ome.jpeg"))

    records = []
    for file_path in tqdm(input_files, desc="Analysing Images and Optimising K"):
        filename = os.path.basename(file_path)
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Failed to load {filename}, skipping.")
            continue

        best_k, spot_counts = find_optimal_k(img)
        counts = [c for _, c in spot_counts]
        variance = np.var(counts)
        weightage = compute_weight(variance)
        plot_k_vs_spots(spot_counts, best_k, filename, output_dir)

        records.append({
            "Image": filename,
            "Optimum_K": best_k,
            "Variance": variance,
            "Weightage": weightage,
        })

    records_df = pd.DataFrame(records)
    records_df["Weights"] = records_df["Weightage"] / records_df["Weightage"].sum()
    records_df.to_csv(os.path.join(output_dir, "k_results.csv"), index=False)
    return records_df


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python KOptimiser2.py <input_dir> <output_dir>")
    else:
        run_k_predictor(sys.argv[1], sys.argv[2])