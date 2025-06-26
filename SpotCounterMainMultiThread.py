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
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

save_overlay = True

def batch_worker(batch_files, output_dir, save_overlay, model_path, scaler_path, probe_type, masks_dir):
    # Only import here, not at module level, to avoid pickling issues
    from core.image_processing import process_images_batch
    return process_images_batch(
        batch_files,
        output_dir,
        save_overlay,
        model_path,
        scaler_path,
        False,  # parallel_features
        masks_dir,
        probe_type
    )

def spot_count(input_dir, output_dir, model_path, scaler_path=None, progress_bar=None, 
               progress_popup=None, root=None, batch_size=64, masks_dir=None, probe_type="BAC", channels=None):
    input_files = glob(os.path.join(input_dir, "*.ome.jpeg")) + \
                  glob(os.path.join(input_dir, "*.tif")) + \
                  glob(os.path.join(input_dir, "*.tiff"))

    # --- Filter files by selected channel (now only one) ---
    if channels:
        # channels is now a string, e.g. "Ch2" or "Ch3"
        ch = str(channels)
        input_files = [f for f in input_files if ch in os.path.splitext(os.path.basename(f))[0]]

    os.makedirs(output_dir, exist_ok=True)
    output = []
    all_k_values = []

    total = len(input_files)
    if progress_bar is not None:
        progress_bar["maximum"] = total
        progress_bar["value"] = 0
        if not hasattr(progress_bar, "label"):
            progress_bar.label = tk.Label(progress_bar.master, text="Estimated time left: --:--", font=("Arial", 9))
            progress_bar.label.pack()

    start_time = time.time()
    processed = 0

    batches = [input_files[i:i+batch_size] for i in range(0, total, batch_size)]

    # --- Only pass simple arguments to the process pool ---
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                batch_worker,
                batch_files,
                output_dir,
                save_overlay,
                model_path,
                scaler_path,
                probe_type,
                masks_dir
            )
            for batch_files in batches
        ]

        for future in as_completed(futures):
            batch_result = future.result()
            # --- GUI objects (progress_bar, root, etc.) are only used here, in the main process ---
            if isinstance(batch_result, tuple) and len(batch_result) == 2:
                batch_results, k_values = batch_result
                all_k_values.extend(k_values)
            else:
                batch_results = batch_result
            output.extend(batch_results)
            processed += batch_size
            if progress_bar is not None:
                progress_bar["value"] = min(processed, total)
                if root is not None:
                    root.update_idletasks()
                elapsed = time.time() - start_time
                if processed > 0:
                    rate = elapsed / processed
                    remaining = (total - processed) * rate
                    mins, secs = divmod(int(remaining), 60)
                    progress_bar.label.config(text=f"Estimated time left: {mins:02d}:{secs:02d}")

    # Final update
    if progress_bar is not None:
        progress_bar["value"] = total
        if root is not None:
            root.update_idletasks()
        # Remove estimated time label after completion
        if hasattr(progress_bar, "label"):
            progress_bar.label.destroy()
            del progress_bar.label
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

def start_counting_popup(root, model_var, scaler_var, model_paths, scaler_paths, mask_var, probe_var, channel_combobox):
    input_dir = filedialog.askdirectory(title="Select Input Folder")
    if not input_dir:
        return
    masks_dir = None
    if mask_var.get() == "Yes":
        masks_dir = filedialog.askdirectory(title="Select Masks Folder")
        if not masks_dir:
            return
    output_dir = filedialog.askdirectory(title="Select Output Folder")
    if not output_dir:
        return
    model_file = model_var.get()
    model_path = model_paths.get(model_file)
    scaler_file = scaler_var.get()
    scaler_path = scaler_paths.get(scaler_file) if scaler_file else None
    if not model_path or not os.path.isfile(model_path):
        messagebox.showerror("Error", "Please select a valid model file.", parent=root)
        return
    if scaler_file and (not scaler_path or not os.path.isfile(scaler_path)):
        messagebox.showerror("Error", "Please select a valid scaler file.", parent=root)
        return

    probe_type = probe_var.get() if hasattr(probe_var, "get") else str(probe_var)

    # --- Get selected channel (single) ---
    selected_channel = channel_combobox.get()
    if not selected_channel:
        messagebox.showerror("Error", "Please select a channel.", parent=root)
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
        spot_count(input_dir, output_dir, model_path, scaler_path, progress_bar, 
                   popup, root, batch_size=64, masks_dir=masks_dir, probe_type=probe_type, channels=selected_channel)
        messagebox.showinfo("Done", "Spot counting completed successfully!", parent=root)
        popup.destroy()
        root.quit()

    threading.Thread(target=run_count, daemon=True).start()

def main():
    root = tk.Tk()
    root.title("Amnis SpotCounter")
    root.geometry("400x380")
    root.resizable(False, False)

    label = tk.Label(root, text="ML-powered spot counting", font=("Arial", 12))
    label.pack(pady=10)

    # Model selection dropdown (show only file names)
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl") and not f.endswith("_scaler.pkl")]
    model_paths = {f: os.path.join(models_dir, f) for f in model_files}
    model_var = tk.StringVar()
    model_label = tk.Label(root, text="Select model:", font=("Arial", 11))
    model_label.pack()
    model_dropdown = ttk.Combobox(root, textvariable=model_var, values=model_files, state="readonly", width=35)
    model_dropdown.pack(pady=5)
    if model_files:
        model_var.set(model_files[0])

    # Scaler selection dropdown (optional)
    scaler_files = [f for f in os.listdir(models_dir) if f.endswith("_scaler.pkl")]
    scaler_paths = {f: os.path.join(models_dir, f) for f in scaler_files}
    scaler_var = tk.StringVar()
    scaler_label = tk.Label(root, text="Select scaler (optional):", font=("Arial", 11))
    scaler_label.pack()
    scaler_dropdown = ttk.Combobox(root, textvariable=scaler_var, values=[""] + scaler_files, state="readonly", width=35)
    scaler_dropdown.pack(pady=5)
    scaler_var.set("")

    # Mask and Probe type toggles (side by side)
    options_frame = tk.Frame(root)
    options_frame.pack(pady=(20, 0))  # Increased top padding

    # Mask toggle
    mask_var = tk.StringVar(value="No")
    mask_label = tk.Label(options_frame, text="Masks?", font=("Arial", 11))
    mask_label.grid(row=0, column=0, padx=(0, 5))
    mask_dropdown = ttk.Combobox(options_frame, textvariable=mask_var, values=["No", "Yes"], state="readonly", width=10)
    mask_dropdown.grid(row=0, column=1, padx=(0, 15))

    # Probe type toggle
    probe_var = tk.StringVar(value="BAC")
    probe_label = tk.Label(options_frame, text="Probe type?", font=("Arial", 11))
    probe_label.grid(row=0, column=2, padx=(0, 5))
    probe_dropdown = ttk.Combobox(options_frame, textvariable=probe_var, values=["BAC", "Oligo"], state="readonly", width=10)
    probe_dropdown.grid(row=0, column=3)

    # --- Channel selector (single-select Combobox) ---
    channel_frame = tk.Frame(root)
    channel_frame.pack(pady=(15, 0))
    channel_label = tk.Label(channel_frame, text="Select channel:", font=("Arial", 11))
    channel_label.pack(side="left", padx=(0, 10))
    channel_combobox = ttk.Combobox(channel_frame, values=["Ch2", "Ch3"], state="readonly", width=8, font=("Arial", 10))
    channel_combobox.set("Ch2")  # Default selection
    channel_combobox.pack(side="left")

    count_button = tk.Button(
        root,
        text="Start counting",
        command=lambda: start_counting_popup(root, model_var, scaler_var, model_paths, scaler_paths, mask_var, probe_var, channel_combobox),
        font=("Arial", 11),
        bg="#4E4CAF",
        fg="white",
        padx=10,
        pady=5
    )
    count_button.pack(pady=(20, 10))  # Increased top padding

    root.mainloop()

if __name__ == '__main__':
    main()