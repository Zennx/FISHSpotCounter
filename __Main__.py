import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import subprocess
import os

APP_TITLE = "ML-powered FISH Spot Counting"

def run_spot_counter():
    try:
        subprocess.Popen(["python", "SpotCounterMainMultiThread.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to launch Spot Counting:\n{e}")

def run_model_trainer():
    try:
        subprocess.Popen(["python", "ModelTrainer.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to launch Model Training:\n{e}")

def main():
    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry("600x350")
    root.resizable(False, False)

    # Main label
    label = tk.Label(root, text=APP_TITLE, font=("Arial", 16, "bold"))
    label.pack(pady=15)

    # Centered frame for buttons
    center_frame = tk.Frame(root)
    center_frame.pack(expand=True)

    btn_spot = tk.Button(
        center_frame, text="Spot Counting", font=("Arial", 13), bg="#4E4CAF", fg="white",
        padx=20, pady=10, width=16, command=run_spot_counter
    )
    btn_spot.pack(pady=(0, 15))

    btn_train = tk.Button(
        center_frame, text="Model Training", font=("Arial", 13), bg="#F38841", fg="white",
        padx=20, pady=10, width=16, command=run_model_trainer
    )
    btn_train.pack(pady=(0, 15))

    # Bottom: Images in a row
    bottom_frame = tk.Frame(root)
    bottom_frame.pack(side="bottom", fill="x", pady=10)

    img_folder = os.path.join(os.getcwd(), "img")
    # try:  # Uncomment if you want to check if the img folder exists
    #    print("Files in img folder:", os.listdir(img_folder))
    #except Exception as e:
    #    print("Could not list img folder:", e)

    uwa_img_path = os.path.join(img_folder, "UWA.jpg")
    iff_img_path = os.path.join(img_folder, "IFF.png")

    def load_img(path, max_height=60, fixed_size=(60, 60), keep_aspect=False):
        try:
            # print(f"Trying to load image: {path}")
            img = Image.open(path)
            if keep_aspect:
                # Resize to fit max_height, keep aspect ratio
                w, h = img.size
                scale = max_height / float(h)
                new_w = int(w * scale)
                img = img.resize((new_w, max_height), Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS)
            else:
                # Resize to fixed square
                img = img.resize(fixed_size, Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Failed to load image {path}: {e}")
            return None

    uwa_img = load_img(uwa_img_path, max_height=60, keep_aspect=True)
    iff_img = load_img(iff_img_path, max_height=70, keep_aspect=True)

    uwa_panel = tk.Frame(bottom_frame)
    uwa_panel.pack(side="left", padx=30, anchor="w")
    if uwa_img:
        uwa_label = tk.Label(uwa_panel, image=uwa_img)
        uwa_label.pack()
    else:
        tk.Label(uwa_panel, text="UWA.jpg\n(not found)", font=("Arial", 9)).pack()

    # Spacer to push IFF logo to the right
    spacer = tk.Frame(bottom_frame)
    spacer.pack(side="left", expand=True, fill="x")

    iff_panel = tk.Frame(bottom_frame)
    iff_panel.pack(side="right", padx=30, anchor="e")
    if iff_img:
        iff_label = tk.Label(iff_panel, image=iff_img)
        iff_label.pack()
    else:
        tk.Label(iff_panel, text="IFF.png\n(not found)", font=("Arial", 9)).pack()

    root.mainloop()

if __name__ == "__main__":
    main()
