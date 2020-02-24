from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import definitions
import preprocessing
import pdb
import datetime
import os

class RandomForestWord():

    def __init__(self, n_train, q_type):
        self.n_train = n_train
        self.q_type = q_type

    def train(self):
        processor = preprocessing.Processor(self.q_type)
        x_train, y_train, x_test, y_test, test_question_copy, vectorizer = processor.random_forest_tfid_char(n_data=self.n_train)

        rnd_clf = RandomForestClassifier(n_estimators=1000,
                                         max_leaf_nodes=100,
                                         n_jobs=-1)

        rnd_clf.fit(x_train, y_train)
        score = rnd_clf.score(x_test, y_test)
        print(score)

        y_pred_rf = rnd_clf.predict(x_test[0:100])
        y_pred_decoded = vectorizer.decode(y_pred_rf)

        dir_results = os.path.join(definitions.ROOT_DIR, "results", "random_forest_char_"+f"{processor.question_type[0:-4]}_"+f"{datetime.datetime.now().strftime('%b-%d')}.txt")
        with open(dir_results, 'w') as file:
            file.write(f'RANDOM FOREST CHARACTER LEVEL MODEL: \t{datetime.datetime.now().strftime("%b-%d-%Y-%H:%M:%S")}\n\tSample Size: {self.n_train}\n')
            file.write(f'Interpolate Test set\n\tScore: {score}\n\nPrediction Sampling\n')

            for pred, y, question in zip(y_pred_decoded, y_test[0:100], test_question_copy[0:100]):
                file.write(f'Input: {question} | {y}\n\t{repr(pred)}\n')
            file.close()


