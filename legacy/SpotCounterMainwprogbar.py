import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk  # Add this import
from glob import glob
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from core.image_processing import process_image

save_overlay=True

def spot_count(input_dir, output_dir, progress_bar=None, progress_popup=None, root=None, update_every=10):
    input_files = glob(os.path.join(input_dir, "*.ome.jpeg")) + \
                  glob(os.path.join(input_dir, "*.tif")) + \
                  glob(os.path.join(input_dir, "*.tiff"))

    os.makedirs(output_dir, exist_ok=True)
    output = []

    total = len(input_files)
    if progress_bar is not None:
        progress_bar["maximum"] = total
        progress_bar["value"] = 0

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, f, output_dir): f for f in input_files}
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                output.append(result)
            completed += 1
            if progress_bar is not None and completed % update_every == 0:
                progress_bar["value"] = completed
                if root is not None:
                    root.update_idletasks()

    # Final update
    if progress_bar is not None:
        progress_bar["value"] = total
        if root is not None:
            root.update_idletasks()
    if progress_popup is not None:
        progress_popup.destroy()

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

def start_counting_popup(root):
    input_dir = filedialog.askdirectory(title="Select Input Folder")
    if not input_dir:
        return
    output_dir = filedialog.askdirectory(title="Select Output Folder")
    if not output_dir:
        return

    # Popup window
    popup = tk.Toplevel(root)
    popup.title("Processing...")
    popup.geometry("350x100")
    popup.resizable(False, False)
    tk.Label(popup, text="Processing images, please wait...", font=("Arial", 11)).pack(pady=10)
    progress_bar = ttk.Progressbar(popup, orient="horizontal", length=300, mode="determinate")
    progress_bar.pack(pady=5)
    popup.grab_set()  # Modal

    def run_count():
        spot_count(input_dir, output_dir, progress_bar, popup, root)
        messagebox.showinfo("Done", "Spot counting completed successfully!", parent=root)
        popup.destroy()
        root.quit()

    threading.Thread(target=run_count, daemon=True).start()

def main():
    root = tk.Tk()
    root.title("Amnis SpotCounter")
    root.geometry("400x200")
    root.resizable(False, False)

    label = tk.Label(root, text="ML-powered spot counting", font=("Arial", 12))
    label.pack(pady=20)

    count_button = tk.Button(
        root,
        text="Start counting",
        command=lambda: start_counting_popup(root),
        font=("Arial", 11),
        bg="#4E4CAF",
        fg="white",
        padx=10,
        pady=5
    )
    count_button.pack(pady=10)

    root.mainloop()

if __name__ == '__main__':
    main()