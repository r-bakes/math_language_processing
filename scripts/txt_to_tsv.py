import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from definitions import DATA_DIR, ROOT_DIR


def process_question_set(difficulty: str, is_train: bool = True) -> None:

    for root, dir, files in os.walk(os.path.abspath(os.path.join(DATA_DIR, "raw", difficulty))):
        for question in files:
            with open(os.path.join(root, question), "r") as f:
                data = f.read()
                data = np.array(data.splitlines()).reshape(-1, 2)
                f.close()

                data = pd.DataFrame({"question": data[:, 0], "answer": data[:, 1]})

                if is_train:
                    train_or_test = "train"
                else:
                    train_or_test = "test"

                data.to_csv(
                    os.path.join(ROOT_DIR, "data", train_or_test, difficulty, question[0:-4])
                    + ".tsv",
                    sep="\t",
                    index=True,
                )


try:
    os.mkdir(os.path.join(ROOT_DIR, "data", "test", "extrapolate"))
    os.mkdir(os.path.join(ROOT_DIR, "data", "test", "interpolate"))
    os.mkdir(os.path.join(ROOT_DIR, "data", "train", "easy"))
    os.mkdir(os.path.join(ROOT_DIR, "data", "train", "medium"))
    os.mkdir(os.path.join(ROOT_DIR, "data", "train", "hard"))
except:
    pass

process_question_set(difficulty='easy')
process_question_set(difficulty='medium')
process_question_set(difficulty='hard')
process_question_set(difficulty='extrapolate', is_train=False)
process_question_set(difficulty='interpolate', is_train=False)