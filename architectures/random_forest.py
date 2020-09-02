from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime
import numpy as np
import os

from definitions import DATA_DIR

def random_forest_experiment(n_train=200000, q_type='algebra__linear_1d.txt'):

    train_data_dir = \
        os.path.join(DATA_DIR, 'train-easy', q_type)


    test_data_dir = \
        os.path.join(DATA_DIR, 'interpolate', q_type)


    with open(train_data_dir, 'r') as f:
        train_data = f.read()
        train_data = np.array(train_data.splitlines()).reshape(-1,2)[0:n_train]

    vectorizer = TfidfVectorizer(analyzer='char', lowercase=True)

    vectorizer.fit(list(train_data[:,0]) + list(train_data[:,1]))

    x = vectorizer.transform(train_data[:,0])
    y = train_data[:,1]

    n_trees = [10, 100, 1000, 5000, 10000]
    for n in n_trees:
        rnd_forest_clf = RandomForestClassifier(n_estimators=n,
                                                random_state=1,
                                                n_jobs=-1,
                                                verbose=0)

        rnd_forest_clf.fit(x,y)

        with open(test_data_dir, 'r') as f:
            test_data = f.read()
            test_data = np.array(test_data.splitlines()).reshape(-1,2)

        x_test = vectorizer.transform(train_data[:,0])

        # score = rnd_forest_clf.score(test_data[0], test_data[1])

        results = rnd_forest_clf.predict(x_test)

        score=0
        for result, solution, question in zip(results, test_data[:,1], test_data[:,0]):
            if result == solution: score+=1

        print(f'With n={n} score was:', score/len(test_data))





random_forest_experiment()






#
#
# class RandomForestWord():
#
#     def __init__(self, n_train, q_type, analysis):
#         self.n_train = n_train
#         self.q_type = q_type
#         self.analysis = analysis
#
#     def train(self):
#         processor = data_preprocessing.Processor(self.q_type)
#         if self.analysis == 'tfid_char':
#             x_train, y_train, x_test, y_test, test_question_copy, vectorizer = processor.tfid_char_preprocess(n_data=self.n_train)
#         elif self.analysis == 'tfid_word':
#             x_train, y_train, x_test, y_test, test_question_copy, vectorizer = processor.tfid_word_preprocess(n_data=self.n_train)
#         elif self.analysis == 'onehot_char':
#             x_train, y_train, x_test, y_test, test_question_copy, vectorizer = processor.onehot_char_preprocess(n_data=self.n_train)
#         elif self.analysis == 'onehot_word':
#             x_train, y_train, x_test, y_test, test_question_copy, vectorizer = processor.onehot_word_preprocess(n_data=self.n_train)
#
#         rnd_clf = RandomForestClassifier(n_estimators=1000,
#                                          n_jobs=-1)
#
#
#         print(f"Fitting on {self.n_train} samples\n")
#         rnd_clf.fit(x_train, y_train)
#
#         score = rnd_clf.score(x_test, y_test)
#
#         print(f"Beginning random sampling and prediction\n")
#         # sample predictions and record for analysis
#         y_pred = rnd_clf.predict(x_test[0:100])
#
#         dir_results = os.path.join(definitions.ROOT_DIR, "results", "random_forest_" + f"{self.analysis}_" + f"{processor.question_type[0:-4]}_" + f"{datetime.datetime.now().strftime('%b-%d')}.txt")
#         with open(dir_results, 'w') as file:
#             file.write(f'RANDOM FOREST CHARACTER LEVEL MODEL: \t{datetime.datetime.now().strftime("%b-%d-%Y-%H:%M:%S")}\n\tSample Size: {self.n_train}\n')
#             file.write(f'Interpolate Test set\n\tScore: {score}\n\nPrediction Sampling\n')
#
#             for pred, y, question in zip(y_pred, y_test[0:100], test_question_copy[0:100]):
#                 file.write(f'Input: {question} | {y}\n\t{repr(pred)}\n')
#             file.close()
#
#
