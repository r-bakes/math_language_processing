from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime
import numpy as np
import os

from definitions import DATA_DIR

def random_forest_experiment(n_train, q_type, analyzer):

    train_data_dir = \
        os.path.join(DATA_DIR, 'train-easy', q_type)


    test_data_dir = \
        os.path.join(DATA_DIR, 'interpolate', q_type)


    with open(train_data_dir, 'r') as f:
        train_data = f.read()
        train_data = np.array(train_data.splitlines()).reshape(-1,2)[0:n_train]

    vectorizer = TfidfVectorizer(analyzer=analyzer, lowercase=True)

    vectorizer.fit(list(train_data[:,0]) + list(train_data[:,1]))

    x = vectorizer.transform(train_data[:,0])
    y = train_data[:,1]


    rnd_forest_clf = RandomForestClassifier(n_estimators=100,
                                            random_state=1,
                                            n_jobs=-1,
                                            verbose=0)

    rnd_forest_clf.fit(x,y)

    with open(test_data_dir, 'r') as f:
        test_data = f.read()
        test_data = np.array(test_data.splitlines()).reshape(-1,2)

    x_test = vectorizer.transform(train_data[:,0])

    results = rnd_forest_clf.predict(x_test)

    score=0
    for result, solution, question in zip(results, test_data[:,1], test_data[:,0]):
        if result == solution: score+=1

    print(f'{q_type}:'.ljust(50), score/len(test_data))
    return q_type, score/len(test_data)

