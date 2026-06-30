import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names=None) -> dict:
    """
    Computes standard hyperspectral image classification metrics:
    Overall Accuracy (OA), Average Accuracy (AA), Cohen's Kappa,
    detailed Classification Report, and Confusion Matrix.
    
    Args:
        y_true (np.ndarray): Ground truth labels (0-indexed).
        y_pred (np.ndarray): Model predictions (0-indexed).
        target_names (list): List of class names.
    Returns:
        dict: Containing 'oa', 'aa', 'kappa', 'report', and 'confusion_matrix'.
    """
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Confusion Matrix to calculate class-wise accuracies
    cm = confusion_matrix(y_true, y_pred)
    
    # Class-wise accuracy = diagonal elements divided by true sample count per class
    with np.errstate(divide='ignore', invalid='ignore'):
        class_acc = np.diagonal(cm) / cm.sum(axis=1)
        # Handle division by zero for classes that might have no true labels in test set
        class_acc = np.nan_to_num(class_acc)
    
    aa = np.mean(class_acc)
    
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    return {
        "oa": oa,
        "aa": aa,
        "kappa": kappa,
        "report": report,
        "confusion_matrix": cm
    }
