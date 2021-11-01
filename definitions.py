import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

MODEL_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
LOG_DIR = os.path.join(ROOT_DIR, "logs", "scalars")

# Data
QUESTION_LIST = [
    filename
    for filename in os.listdir(os.path.join(ROOT_DIR, "data", "train", "easy"))
    if filename.endswith(".tsv")
]
