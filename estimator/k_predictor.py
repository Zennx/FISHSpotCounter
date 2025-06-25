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

from core.image_processing import extract_features, analyze_image_for_spots

EXPECTED_SPOTS = 2
K_RANGE = (0.5, 3.0)
K_STEP = 0.05
K_FINE_STEP = 0.01


def count_spots_at_k(img, features, k, probe_type='oligo', noise_matrix=None):
    # Use the centralized analysis pipeline for spot counting
    return analyze_image_for_spots(img, features, k, probe_type=probe_type, noise_matrix=noise_matrix)

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


def get_k_range(probe_type):
    probe_type_str = str(probe_type).strip().lower()
    if probe_type_str == 'bac':
        return (1.0, 12.0)
    else:
        return (0.5, 3.0)


def find_optimal_k(img, probe_type='oligo'):
    features = extract_features(img)  # Extract once
    k_range = get_k_range(probe_type)
    ks = np.arange(k_range[0], k_range[1] + K_STEP, K_STEP)
    # Generate a noise matrix for this image size (optional, or pass None)
    h, w = img.shape[:2]
    noise_matrix = np.random.normal(loc=0, scale=0.01, size=(h, w)).astype(np.float32)
    spot_counts = [(k, count_spots_at_k(img, features, k, probe_type=probe_type, noise_matrix=noise_matrix)) for k in ks]

    best_k = find_longest_contiguous_k(spot_counts)

    if best_k is None:
        # Try finer resolution if no valid K found
        ks_fine = np.arange(k_range[0], k_range[1] + K_FINE_STEP, K_FINE_STEP)
        spot_counts = [(k, count_spots_at_k(img, features, k, probe_type=probe_type, noise_matrix=noise_matrix)) for k in ks_fine]
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


def run_k_predictor(input_dir, output_dir, probe_type='oligo'):
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

        best_k, spot_counts = find_optimal_k(img, probe_type=probe_type)
        counts = [c for _, c in spot_counts]
        variance = np.var(counts)
        plot_k_vs_spots(spot_counts, best_k, filename, output_dir)

        records.append({
            "Image": filename,
            "Optimum_K": best_k,
            "Variance": variance,
        })

    records_df = pd.DataFrame(records)

    # New weighting: inverse frequency of Optimum_K (rounded for grouping)
    if not records_df.empty:
        # Round K to 3 decimals for grouping (adjust as needed)
        k_rounded = records_df["Optimum_K"].round(3)
        freq = k_rounded.value_counts()
        # Assign weight = 1/frequency for each image
        records_df["Weightage"] = k_rounded.map(lambda k: 1.0 / freq[k])
        # Normalise weights to sum to 1
        records_df["Weights"] = records_df["Weightage"] / records_df["Weightage"].sum()
    else:
        records_df["Weights"] = 1.0

    records_df.to_csv(os.path.join(output_dir, "k_results.csv"), index=False)
    return records_df


# GUI integration note:
# To add a probe type switch in your GUI, add a dropdown or radio button for 'Oligo'/'BAC'.
# Pass the selected value to run_k_predictor(..., probe_type=selected_probe_type)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print("Usage: python k_predictor.py <input_dir> <output_dir> <probe_type>")
    else:
        run_k_predictor(sys.argv[1], sys.argv[2], probe_type=sys.argv[3])