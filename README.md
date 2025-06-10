# FISHSpotCounter
High throughput single-cell analysis pipeline with ML-powered adaptive k value thresholding for spot counting of FISH spots from Amnis true imaging flow cytometers
___________________________________________________________________________________
INTRODUCTION
What is the FISHSpotCounter project?




Running the application: Where to start?


TO BE ADDED IN THE FUTURE

___________________________________________________________________________________
File layout:
project_root/
│
├── core/
│   ├── image_processing.py      # Filtering, thresholding, pre/post-processing
│   ├── spot_counter.py          # `detect_spots` logic
│   └── file_handler.py          # File loaders, normalisation, etc.
│
├── estimator/
│   └── k_predictor.py           # Load model and batch-predict K values
│
├── optimiser/
│   └── k_optimiser.py           # Recursive/bimodal-aware optimiser for model training
│
├── training/
│   ├── feature_extraction.py    # Converts image + K into features
│   └── model_training.py        # Trains ML models such as XGBoost (default) or similar
│
├── gui/
│   └── SpotCounterGUI.py        # GUI frontend using tkinter, PyQt, etc. for the app
│
├── main.py                      # Entrypoint with argument parsing
├── config.py
└── requirements.txt             # Dependencies list, PIP install -r compatible

Other files:
README.md               --> opens this readme file!
DevelopmentNotes.md     --> a record and notes during the development of this app
.gitattributes          --> hidden file for git version control