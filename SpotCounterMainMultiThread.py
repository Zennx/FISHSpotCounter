import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from glob import glob
import pandas as pd
import threading

from core.image_processing import process_images_batch

save_overlay = True

def spot_count(input_dir, output_dir, model_path, progress_bar=None, progress_popup=None, root=None, update_every=5, batch_size=16):
    input_files = glob(os.path.join(input_dir, "*.ome.jpeg")) + \
                  glob(os.path.join(input_dir, "*.tif")) + \
                  glob(os.path.join(input_dir, "*.tiff"))

    os.makedirs(output_dir, exist_ok=True)
    output = []

    total = len(input_files)
    if progress_bar is not None:
        progress_bar["maximum"] = total
        progress_bar["value"] = 0

    # Split files into batches
    for i in range(0, total, batch_size):
        batch_files = input_files[i:i+batch_size]
        batch_results = process_images_batch(batch_files, output_dir, save_overlay=save_overlay, model_path=model_path)
        output.extend(batch_results)
        if progress_bar is not None:
            progress_bar["value"] = min(i + batch_size, total)
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
        meanSpotCount = f"{np.mean(df['SpotCount']):.3f}" if "SpotCount" in df.columns else "N/A"
        if not frequency.empty:
            plt.bar(frequency.index, frequency.values, color='skyblue')
            for i, (x, v) in enumerate(zip(frequency.index, frequency.values)):
                plt.text(x, v + 0.2, str(v), ha='center', va='bottom', fontsize=10)
            plt.xlabel('Spot Count')
            plt.ylabel('Frequency')
            plt.title('Spot Count Frequency - Weighted Regression Model')
            plt.text(0.7, 0.9, 'Mean Spot Count = ' + meanSpotCount, fontsize=8, transform=plt.gca().transAxes)
            plt.xticks(np.arange(min(frequency.index), max(frequency.index)+1, 1))  # fix tic spacing
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "spot_counts.png"))
            plt.close()
        else:
            print("Warning: No spot counts to plot.")
    else:
        print("Warning: 'SpotCount' not found in DataFrame.")
    print("Spot counting complete.")
    print("Results saved to " + output_dir + " as 'spot_counts.csv'.")
    if "frequency" in locals():
        print(frequency)
    print("Mean Spot Count:" + meanSpotCount)

# GUI setup

def start_counting_popup(root, model_var):
    input_dir = filedialog.askdirectory(title="Select Input Folder")
    if not input_dir:
        return
    output_dir = filedialog.askdirectory(title="Select Output Folder")
    if not output_dir:
        return
    model_path = model_var.get()
    if not model_path or not os.path.isfile(model_path):
        messagebox.showerror("Error", "Please select a valid model file.", parent=root)
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
        spot_count(input_dir, output_dir, model_path, progress_bar, popup, root)
        messagebox.showinfo("Done", "Spot counting completed successfully!", parent=root)
        popup.destroy()
        root.quit()

    threading.Thread(target=run_count, daemon=True).start()

def main():
    root = tk.Tk()
    root.title("Amnis SpotCounter")
    root.geometry("400x260")
    root.resizable(False, False)

    label = tk.Label(root, text="ML-powered spot counting", font=("Arial", 12))
    label.pack(pady=10)

    # Model selection dropdown (show only file names)
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
    model_paths = {f: os.path.join(models_dir, f) for f in model_files}
    model_var = tk.StringVar()
    model_label = tk.Label(root, text="Select model:", font=("Arial", 11))
    model_label.pack()
    model_dropdown = ttk.Combobox(root, textvariable=model_var, values=model_files, state="readonly", width=35)
    model_dropdown.pack(pady=5)
    if model_files:
        model_var.set(model_files[0])

    def start_counting_popup_with_path(root, model_var):
        input_dir = filedialog.askdirectory(title="Select Input Folder")
        if not input_dir:
            return
        output_dir = filedialog.askdirectory(title="Select Output Folder")
        if not output_dir:
            return
        model_file = model_var.get()
        model_path = model_paths.get(model_file)
        if not model_path or not os.path.isfile(model_path):
            messagebox.showerror("Error", "Please select a valid model file.", parent=root)
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
            spot_count(input_dir, output_dir, model_path, progress_bar, popup, root)
            messagebox.showinfo("Done", "Spot counting completed successfully!", parent=root)
            popup.destroy()
            root.quit()

        threading.Thread(target=run_count, daemon=True).start()

    count_button = tk.Button(
        root,
        text="Start counting",
        command=lambda: start_counting_popup_with_path(root, model_var),
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