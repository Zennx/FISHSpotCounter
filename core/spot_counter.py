from skimage import feature, draw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

def detect_spots_log(image, min_sigma=1.25, max_sigma=5, num_sigma=10, threshold=0.05):
    """Detect spots using Laplacian of Gaussian (LoG) for oligo probes."""
    blobs = feature.blob_log(image, min_sigma=min_sigma, max_sigma=max_sigma,
                             num_sigma=num_sigma, threshold=threshold)
    # Compute radii
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    # print(f"Detected {len(blobs)} spots using LoG")
    return blobs

def detect_spots_log_bac(image, min_sigma=1, max_sigma=12, num_sigma=14, threshold=0.3, overlap=0.8):
    """Detect spots using Laplacian of Gaussian (LoG) for BAC probes."""
    blobs_raw = feature.blob_log(image, min_sigma=min_sigma, max_sigma=max_sigma,
                             num_sigma=num_sigma, threshold_rel=threshold, overlap=overlap)
    # Compute radii
    blobs_raw[:, 2] = blobs_raw[:, 2] * np.sqrt(2)
    blobs = [b for b in blobs_raw if b[2] > 1 and b[2] < 8]  # Filter out too small or too large blobs
    # print(f"Detected {len(blobs)} spots using DoH")
    return blobs

def detect_spots_log2(image, min_sigma=1, max_sigma=10, num_sigma=12, threshold=0.35, overlap=0.7):
    """Detect spots using Determinant of Hessian (DoH) for BAC probes."""
    blobs_raw = feature.blob_doh(image, min_sigma=min_sigma, max_sigma=max_sigma,
                             num_sigma=num_sigma, threshold=threshold, overlap=overlap)
    # Compute radii
    blobs_raw[:, 2] = blobs_raw[:, 2] * np.sqrt(2)
    blobs = [b for b in blobs_raw if b[2] > 1 and b[2] < 8]  # Filter out too small or too large blobs
    # print(f"Detected {len(blobs)} spots using DoH")
    return blobs

def detect_spots_dohX(image, min_distance=2, threshold_abs=0.5, max_radius=8, threshold_ratio=0.5, overlap_thresh=0.8):
    """
    Detect spots using local maxima after LoG filtering.
    Returns an array of [y, x, r] for each detected spot.
    """
    # Apply Laplacian of Gaussian filter to enhance spots
    # filtered = gaussian_laplace(image, sigma=sigma)
    # Invert because LoG gives negative peaks for bright spots
    # filtered = -filtered
    # Detect local maxima
    coordinates = feature.peak_local_max(
        image,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        exclude_border=False
    )
    # Assign a fixed radius (can be tuned or estimated)
    radii = []

    for y, x in coordinates:
        peak_intensity = image[y, x]
        found_radius = max_radius

        for r in range(1, max_radius):
            # Create circular ring mask at radius r
            rr, cc = draw.circle_perimeter(y, x, r, image.shape)
            ring_values = image[rr, cc]

            if np.mean(ring_values) < threshold_ratio * peak_intensity:
                found_radius = r
                break

        radii.append(found_radius)

    kept_coords = []
    kept_radii = []

    for i, (cyx, r) in enumerate(zip(coordinates, radii)):
        keep = True
        for j in range(len(kept_coords)):
            dist = np.linalg.norm(cyx - kept_coords[j])
            r_sum = r + kept_radii[j]
            if dist < overlap_thresh * r_sum:
                keep = False
                break
        if keep:
            kept_coords.append(cyx)
            kept_radii.append(r)

    spots = np.array([[y, x, radii] for y, x in kept_coords])
    return spots

def save_spot_overlay(image, blobs, save_path, title="LoG Spot Detection"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray')
    for y, x, r in blobs:
        circle = patches.Circle((x, y), r, color='red', linewidth=1.5, fill=False)
        ax.add_patch(circle)
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()