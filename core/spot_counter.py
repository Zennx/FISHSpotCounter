from skimage import feature
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def detect_spots_log(image, min_sigma=1.25, max_sigma=5, num_sigma=10, threshold=0.01):
    """Detect spots using Laplacian of Gaussian (LoG)."""
    blobs = feature.blob_log(image, min_sigma=min_sigma, max_sigma=max_sigma,
                             num_sigma=num_sigma, threshold=threshold)
    # Compute radii
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    return blobs

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