
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

def map_topics_to_labels(topic_assignments, true_labels):
    """
    Maps each cluster (topic) to the most frequent true label in that cluster.
    Returns:
        mapping: dict {topic_id: label}
        y_pred_mapped: list of predicted labels mapped from topics
    """
    df = pd.DataFrame({'topic': topic_assignments, 'label': true_labels})
    
    # Calculate contingency matrix
    contingency = pd.crosstab(df['topic'], df['label'])
    
    # Simple mapping: For each topic, assign the label that appears most frequently
    mapping = {}
    for topic in contingency.index:
        mapping[topic] = contingency.loc[topic].idxmax()
        
    y_pred_mapped = [mapping[t] for t in topic_assignments]
    return mapping, y_pred_mapped

def compute_supervised_metrics(topic_assignments, true_labels):
    """
    Computes Accuracy, Precision, Recall, F1, and Confusion Matrix.
    """
    # Remove rows where true_label is NaN
    mask = pd.notna(true_labels)
    clean_topics = np.array(topic_assignments)[mask]
    clean_labels = np.array(true_labels)[mask]
    
    # Map topics to labels
    mapping, y_pred = map_topics_to_labels(clean_topics, clean_labels)
    
    # Compute metrics
    acc = accuracy_score(clean_labels, y_pred)
    prec = precision_score(clean_labels, y_pred, average='weighted', zero_division=0)
    rec = recall_score(clean_labels, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(clean_labels, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1
    }
    
    return metrics, clean_labels, y_pred, mapping

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """
    Plots confusion matrix.
    """
    # Get unique labels from both true and pred to ensure matrix shape is correct
    labels = sorted(list(set(y_true) | set(y_pred)))
    
    # Limit to top N labels if too many
    if len(labels) > 20:
        print(f"[INFO] Too many labels ({len(labels)}). Plotting matrix for top 20 most frequent.")
        top_labels = pd.Series(y_true).value_counts().nlargest(20).index.tolist()
        # Filter data
        mask = pd.Series(y_true).isin(top_labels) & pd.Series(y_pred).isin(top_labels)
        y_true = np.array(y_true)[mask]
        y_pred = np.array(y_pred)[mask]
        labels = top_labels

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label (Mapped from Topic)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

