from typing import Dict, List, Union
import numpy as np

def calculate_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    num_classes: int = 3
) -> Dict[str, Union[float, List[float]]]:
    """
    Computes classification metrics using pure NumPy (zero external sklearn dependency).

    Metrics:
      - accuracy
      - macro_f1
      - precision (per class)
      - recall (per class)
      - f1 (per class)
    """
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)

    f1_scores = []
    class_precisions = []
    class_recalls = []

    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        precision = float(tp / (tp + fp + 1e-9))
        recall = float(tp / (tp + fn + 1e-9))
        f1 = float(2 * (precision * recall) / (precision + recall + 1e-9))

        class_precisions.append(precision)
        class_recalls.append(recall)
        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores))
    accuracy = float(np.mean(y_true == y_pred))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "precision": class_precisions,
        "recall": class_recalls,
        "f1": f1_scores
    }
