from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


def random_forest_pipeline(
    analyzer: str, max_depth: int, number_estimators: int,
) -> Pipeline:
    """

    Args:
        analyzer:
        max_depth:
        number_estimators:

    Returns:

    """

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer=analyzer, lowercase=True)),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=number_estimators,
                    random_state=1,
                    n_jobs=-1,
                    verbose=1,
                    max_depth=max_depth,
                ),
            ),
        ]
    )

    return pipeline
