import threading
import estimator.k_predictor
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import cv2
import os
from core.image_processing import extract_features
from training.model_training import xgbost_train
import glob

def extract_and_save_features(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    feature_records = []
    image_files = []
    for ext in ("*.tif", "*.tiff", "*.ome.jpeg"):
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    for file_path in image_files:
        filename = os.path.basename(file_path)
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Failed to read {filename}, skipping.")
            continue
        features = extract_features(img)
        feature_records.append({"Image": filename, **{f"F{i+1}": v for i, v in enumerate(features)}})
    features_df = pd.DataFrame(feature_records)
    optimiser_features_csv = os.path.join(output_dir, "features.csv")
    features_df.to_csv(optimiser_features_csv, index=False)
    print(f"Features saved to {optimiser_features_csv}")
    return features_df  # Return the DataFrame directly

def run_optimiser_and_train(input_dir, output_dir, model_name, probe_type, root):
    features_df = extract_and_save_features(input_dir, output_dir)
    results_df = estimator.k_predictor.run_k_predictor(input_dir, output_dir, probe_type=probe_type)
    # Ensure models directory exists
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_name + ".pkl")
    xgbost_train(features_df, results_df, model_path)
    messagebox.showinfo("Done", f"Optimisation and training successful!\nModel saved as:\n{model_path}", parent=root)
    root.quit()

def start_optimising_popup(root, model_name_var, probe_type_var):
    input_dir = filedialog.askdirectory(title="Select Input Folder")
    if not input_dir:
        return
    output_dir = filedialog.askdirectory(title="Select Output Folder")
    if not output_dir:
        return
    model_name = model_name_var.get().strip()
    if not model_name:
        messagebox.showerror("Error", "Please enter a model name.", parent=root)
        return
    probe_type = probe_type_var.get()
    threading.Thread(target=run_optimiser_and_train, args=(input_dir, output_dir, model_name, probe_type, root), daemon=True).start()

def main():
    root = tk.Tk()
    root.title("Amnis SpotCounter - Model Training")
    root.geometry("400x300")
    root.resizable(False, False)

    label = tk.Label(root, text="Model training using XGBoost.", font=("Arial", 12))
    label2 = tk.Label(root, text="Input true 2-spots images.", font=("Arial", 12))
    label.pack(pady=10)
    label2.pack(pady=10)

    # Model name entry
    model_name_var = tk.StringVar()
    model_name_label = tk.Label(root, text="Model name:", font=("Arial", 11))
    model_name_label.pack()
    model_name_entry = tk.Entry(root, textvariable=model_name_var, font=("Arial", 11), width=30)
    model_name_entry.pack(pady=5)
    model_name_var.set("k_predictor_model")  # Default value

    # Probe type dropdown
    probe_type_var = tk.StringVar(value="Oligo")
    probe_type_label = tk.Label(root, text="Probe type:", font=("Arial", 11))
    probe_type_label.pack()
    probe_type_dropdown = tk.OptionMenu(root, probe_type_var, "Oligo", "BAC")
    probe_type_dropdown.config(width=10, font=("Arial", 11))
    probe_type_dropdown.pack(pady=5)

    count_button = tk.Button(
        root,
        text="Train XGBoost Model",
        command=lambda: start_optimising_popup(root, model_name_var, probe_type_var),
        font=("Arial", 11),
        bg="#F38841",
        fg="white",
        padx=10,
        pady=5
    )
    count_button.pack(pady=10)

    root.mainloop()

if __name__ == '__main__':
    main()