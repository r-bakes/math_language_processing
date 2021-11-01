"""
Contains the primary wrapper for executing training of a random forest given a question type and difficulty.
"""
from joblib import dump
from datetime import datetime
import os

from definitions import MODEL_DIR
from src.data_preprocessing import get_data
from src.helpers import score_results
from .random_forest import random_forest_pipeline


def random_forest_experiment(
    experiment_name: str,
    train_set_size: int,
    difficulty: str,
    question_type: str,
    analyzer: str,
    max_depth: int,
    number_estimators: int,
    save: bool,
) -> None:
    """Primary wrapper for the execution of a random forest classifier on a given question type.

    Args:
        experiment_name:
        train_set_size:
        difficulty:
        question_type:
        analyzer:
        max_depth:
        number_estimators:
        save:

    Returns:

    """

    x_train, y_train = get_data(
        difficulty=difficulty, question_type=question_type, set_size=train_set_size
    )

    pipeline = random_forest_pipeline(
        analyzer=analyzer, max_depth=max_depth, number_estimators=number_estimators
    )

    pipeline.fit(x_train, y_train)

    if save:
        dump(
            pipeline,
            os.path.join(
                MODEL_DIR, "random_forests", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{experiment_name}"
            ),
        )

    x_test, y_test = get_data(question_type=question_type)

    predictions = pipeline.predict(x_test)

    score_results(y=y_test, predictions=predictions, question_type=question_type)
