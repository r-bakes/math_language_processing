"""
This file contains helper functions universal to all algorithms experimented with.
"""

def score_results(
        y: list,
        predictions: list,
        question_type: str
) -> dict:
    """

    Args:
        y: true target labels.
        predictions: model predicted labels.
        question_type: question type model was trained on.

    Returns: dictionary with final score and question.

    """

    score = 0

    for solution, prediction in zip(y, predictions):
        if solution == prediction:
            score += 1

    final_score = {
        'question': question_type,
        'score': round(score / len(predictions), 2)*100
    }

    print(f"{question_type} Score:".ljust(50), f"{final_score['score']}%")

    return final_score
