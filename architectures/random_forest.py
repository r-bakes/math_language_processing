from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import definitions
import numpy as np
import preprocessing
import pdb
import datetime
import os

class RandomForestWord():

    def __init__(self, n_train, q_type, analysis):
        self.n_train = n_train
        self.q_type = q_type
        self.analysis = analysis

    def train(self):
        processor = preprocessing.Processor(self.q_type)
        if self.analysis == 'tfid_char':
            x_train, y_train, x_test, y_test, test_question_copy, vectorizer = processor.tfid_char_preprocess(n_data=self.n_train)
        elif self.analysis == 'tfid_word':
            x_train, y_train, x_test, y_test, test_question_copy, vectorizer = processor.tfid_word_preprocess(n_data=self.n_train)
        elif self.analysis == 'onehot_char':
            x_train, y_train, x_test, y_test, test_question_copy, vectorizer = processor.onehot_char_preprocess(n_data=self.n_train)
        elif self.analysis == 'onehot_word':
            x_train, y_train, x_test, y_test, test_question_copy, vectorizer = processor.onehot_word_preprocess(n_data=self.n_train)

        rnd_clf = RandomForestClassifier(n_estimators=1000,
                                         max_leaf_nodes=100,
                                         n_jobs=-1)


        print(f"Fitting on {self.n_train} samples\n")
        rnd_clf.fit(x_train, y_train)

        pdb.set_trace()
        score = rnd_clf.score(x_test, y_test)

        print(f"Beginning random sampling and prediction\n")
        # sample predictions and record for analysis
        y_pred = rnd_clf.predict(x_test[0:100])

        dir_results = os.path.join(definitions.ROOT_DIR, "results", "random_forest_" + f"{self.analysis}_"+f"{processor.question_type[0:-4]}_"+f"{datetime.datetime.now().strftime('%b-%d')}.txt")
        with open(dir_results, 'w') as file:
            file.write(f'RANDOM FOREST CHARACTER LEVEL MODEL: \t{datetime.datetime.now().strftime("%b-%d-%Y-%H:%M:%S")}\n\tSample Size: {self.n_train}\n')
            file.write(f'Interpolate Test set\n\tScore: {score}\n\nPrediction Sampling\n')

            for pred, y, question in zip(y_pred, y_test[0:100], test_question_copy[0:100]):
                file.write(f'Input: {question} | {y}\n\t{repr(pred)}\n')
            file.close()


