
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

def plot_top_words(texts, title="Top 20 Most Frequent Words", n=20, save_path=None):
    """
    Plots a bar chart of the top N most frequent words.
    """
    all_words = ' '.join(texts).split()
    word_counts = Counter(all_words)
    common_words = word_counts.most_common(n)
    
    words = [w[0] for w in common_words]
    counts = [w[1] for w in common_words]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=words, palette='viridis')
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_wordcloud(texts, title="Word Cloud", save_path=None):
    """
    Generates and plots a Word Cloud.
    """
    text_combined = ' '.join(texts)
    
    # Check if text is empty
    if not text_combined.strip():
        print("No text available for WordCloud.")
        return

    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(text_combined)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved wordcloud to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_category_distribution(df, column='tag1', title="News Category Distribution", save_path=None):
    """
    Plots the count of each category.
    """
    plt.figure(figsize=(12, 8))
    # Get top 20 categories if too many
    top_cats = df[column].value_counts().head(20)
    sns.barplot(x=top_cats.values, y=top_cats.index, palette='magma')
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("Category")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_text_length_distribution(df, raw_col='Content', clean_col='clean_content', save_path=None):
    """
    Plots histograms of text lengths before and after processing.
    """
    raw_lens = df[raw_col].astype(str).apply(len)
    clean_lens = df[clean_col].astype(str).apply(len)
    
    plt.figure(figsize=(10, 6))
    plt.hist(raw_lens, bins=50, alpha=0.5, label='Original Length', color='blue', range=(0, 5000))
    plt.hist(clean_lens, bins=50, alpha=0.5, label='Processed Length', color='green', range=(0, 5000))
    plt.title("Text Length Distribution (Before vs After)")
    plt.xlabel("Character Count")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_missing_values_overall(df, title="Overall Data Completeness", save_path=None):
    """
    Plots a simple Pie Chart showing Total Present vs Total Missing values in the entire dataset.
    """
    total_cells = df.size
    total_missing = df.isnull().sum().sum()
    total_present = total_cells - total_missing
    
    if total_missing == 0:
        print("No missing values found in the entire dataset.")
        # Save a placeholder
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "Dataset is 100% Complete (No Missing Values)", ha='center', va='center')
        plt.axis('off')
        if save_path:
            plt.savefig(save_path)
        plt.close()
        return

    data = [total_present, total_missing]
    labels = [f'Present ({total_present})', f'Missing ({total_missing})']
    colors = ['#66b3ff', '#ff9999']

    plt.figure(figsize=(8, 6))
    plt.pie(data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(title)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_missing_values_per_column(df, title="Missing Values per Column", save_path=None):
    """
    Plots missing values count per column (Bar Chart).
    """
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    
    if null_counts.empty:
        # If no missing values, we do nothing or save placeholder (handled by overall usually)
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x=null_counts.index, y=null_counts.values, palette='Reds')
    plt.title(title)
    plt.ylabel("Count of Missing Values")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
