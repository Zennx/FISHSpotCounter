import numpy as np
from skimage import data, exposure, img_as_float, io, filters, morphology, feature, util, measure
from skimage.feature import blob_doh
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from glob import glob
import pandas as pd
import cv2
from tqdm.tk import tqdm, trange
from sklearn.ensemble import RandomForestRegressor
import joblib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from core.image_processing import process_image
from core.spot_counter import detect_spots_log, save_spot_overlay

# Load the pre-trained model
save_overlay=True
model = joblib.load("C:/Users/ONG32/trained_model_XGB_V3.1.pkl")
    
def spot_count(input_dir, output_dir):
    input_files = glob(os.path.join(input_dir, "*.ome.jpeg")) + \
                  glob(os.path.join(input_dir, "*.tif")) + \
                  glob(os.path.join(input_dir, "*.tiff"))

    os.makedirs(output_dir, exist_ok=True)
    output = []

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, f, output_dir): f for f in input_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            result = future.result()
            if result is not None:
                output.append(result)

    # Create DataFrame and save results
    df = pd.DataFrame(output)
    df.to_csv(os.path.join(output_dir, "spot_results.csv"), index=False)

    # Now this will work (assuming SpotCount column is present)
    if "SpotCount" in df.columns:
        frequency = df["SpotCount"].value_counts().sort_index()
        plt.bar(frequency.index, frequency.values, color='skyblue')
        for i, (x, v) in enumerate(zip(frequency.index, frequency.values)):
            plt.text(x, v + 0.2, str(v), ha='center', va='bottom', fontsize=10)
        plt.xlabel('Spot Count')
        plt.ylabel('Frequency')
        plt.title('Spot Count Frequency - Weighted Regression Model')
        plt.xticks(np.arange(min(frequency.index), max(frequency.index)+1, 1))  # fix tic spacing
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "spot_counts.png"))
    else:
        print("Warning: 'SpotCount' not found in DataFrame.")
    print("Spot counting complete.")
    print("Results saved to " + output_dir + " as 'spot_counts.csv'.")
    print(frequency)
    meanSpotCount = str(f"{np.mean(df["SpotCount"]):.3f}")
    print("Mean Spot Count:" + meanSpotCount)

# GUI setup

def start_counting():
    input_dir = filedialog.askdirectory(title="Select Input Folder")
    if not input_dir:
        return
    output_dir = filedialog.askdirectory(title="Select Output Folder")
    if not output_dir:
        return

    spot_count(input_dir, output_dir)
    messagebox.showinfo("Done", "Spot counting completed successfully!")
    exit()

def main():
    root = tk.Tk()
    root.title("Amnis SpotCounter")
    root.geometry("400x200")
    root.resizable(False, False)

    label = tk.Label(root, text="ML-powered spot counting", font=("Arial", 12))
    label.pack(pady=20)

    count_button = tk.Button(root, text="Start counting", command=start_counting, font=("Arial", 11), bg="#4CAF50", fg="white", padx=10, pady=5)
    count_button.pack(pady=10)

    root.mainloop()

if __name__ == '__main__':
    main()