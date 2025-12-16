
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

def plot_coherence_per_topic(coherence_list, title="Coherence Score per Topic", save_path=None):
    """
    Plots a bar chart of coherence score for each topic.
    """
    topics = range(1, len(coherence_list) + 1)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(topics), y=coherence_list, palette='coolwarm')
    plt.title(title)
    plt.xlabel("Topic #")
    plt.ylabel("Coherence Score ($C_v$)")
    plt.axhline(y=np.mean(coherence_list), color='red', linestyle='--', label=f'Mean: {np.mean(coherence_list):.4f}')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

from sklearn.metrics import classification_report

def plot_class_f1_heatmap(y_true, y_pred, title="Class-wise F1 Scores (Top 20)", save_path=None):
    """
    Plots a heatmap/bar of F1 scores for the top 20 most frequent classes.
    """
    # Get top 20 labels
    top_labels = pd.Series(y_true).value_counts().nlargest(20).index.tolist()
    
    # Compute report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Extract f1-scores for top labels
    f1_scores = {}
    for label in top_labels:
        if label in report:
            f1_scores[label] = report[label]['f1-score']
        else:
            f1_scores[label] = 0.0
            
    # Convert to DF for plotting
    df_f1 = pd.DataFrame(list(f1_scores.items()), columns=['Class', 'F1 Score'])
    df_f1 = df_f1.sort_values('F1 Score', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='F1 Score', y='Class', data=df_f1, palette='viridis')
    plt.title(title)
    plt.xlim(0, 1)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

