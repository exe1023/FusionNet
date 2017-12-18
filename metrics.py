from collections import Counter
import numpy as np
def f1_score(start, end, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    prediction_tokens = [x for x in range(start, end)]
    ground_truth_tokens = [x for x in range(ground_truth[0], ground_truth[1])]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(start, end, ground_truth):
    """Check if the prediction is a (soft) exact match with the ground truth."""
    return (start == ground_truth[0] and end == ground_truth[1])

def batch_score(starts, ends, answers):
    assert len(starts) == len(answers)
    f1_scores = []
    exact_match_scores = []
    for i in range(len(answers)):
        f1_scores.append(f1_score(starts[i][0], ends[i][0], answers[i]))
        exact_match_scores.append(exact_match_score(starts[i][0], ends[i][0], answers[i]))
    return np.mean(f1_scores), np.mean(exact_match_scores)
