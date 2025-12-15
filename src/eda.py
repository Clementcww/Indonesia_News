
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
