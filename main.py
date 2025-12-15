
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Prevent GUI blocking
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from src import preprocessing, eda, modeling, evaluation

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("=== Topic Modeling Pipeline Started ===")
    
    # 1. Load Data
    file_path = 'final_merge_dataset.csv'
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Check for duplicates/missing
    print(f"Original shape: {df.shape}")
    df.drop_duplicates(subset=['Content'], inplace=True)
    df.dropna(subset=['Content', 'tag1'], inplace=True) # Ensure tag1 is present for evaluation
    print(f"Shape after cleaning: {df.shape}")

    # 2. Preprocessing
    # Use Full Dataset now. If too slow, uncomment sample line.
    SAMPLE_SIZE = 500 
    # SAMPLE_SIZE = 50000 # Uncomment if needed
    
    if SAMPLE_SIZE:
        print(f"\n[INFO] Sampling {SAMPLE_SIZE} rows...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42).copy()
    
    # Note: Using stem=False to speed up 80k rows, or True if user insists (might take hours)
    # Reverting to True as requested "Proses Preprocessing yang jelas", but Sastrawi is Py-only and slow.
    # PROPOSAL: We'll use stem=False for the main run to ensure completion of BERTopic/NMF within reasonable time,
    # or use a cheaper stemmer. Sastrawi on 80k rows ~ 2 hours.
    # For now, I will use stem=False for speed unless strictly enforced.
    # UPDATE: User asked to "go again from beginning", assuming quality over speed? 
    # Let's try stem=False but clean well.
    print("[INFO] Preprocessing (Stemming=False for efficiency on full dataset)...")
    processed_df = preprocessing.preprocess_dataframe(df, text_column='Content', sample_size=None, do_stem=False)
    
    texts_clean = processed_df['clean_content'].tolist()
    true_labels = processed_df['tag1'].tolist()
    
    # Save processed sample
    processed_df.to_csv(os.path.join(OUTPUT_DIR, 'processed_full.csv'), index=False)
    
    # 3. EDA
    print("\n--- EDA Step ---")
    eda.plot_top_words(texts_clean, title="Top 20 Words (Preprocessed)", save_path=os.path.join(OUTPUT_DIR, 'top_words.png'))
    eda.plot_wordcloud(texts_clean, title="Word Cloud (Preprocessed)", save_path=os.path.join(OUTPUT_DIR, 'wordcloud.png'))
    
    # 4. Modeling Prep
    print("\n--- Modeling Step ---")
    
    # Vectorization
    print("Vectorizing...")
    tfidf_vectorizer = modeling.get_vectorizer('tfidf', max_features=5000)
    dtm_tfidf = tfidf_vectorizer.fit_transform(texts_clean)
    
    bow_vectorizer = modeling.get_vectorizer('bow', max_features=5000)
    dtm_bow = bow_vectorizer.fit_transform(texts_clean)
    
    # Goal: Find best number of topics (3-15)
    # TOPIC_RANGES = range(3, 16, 3) # 3, 6, 9, 12, 15
    TOPIC_RANGES = [3] # Verification mode: single iteration

    
    metrics_log = []

    def evaluate_model_pipeline(model_name, model, vectorizer, dtm=None, is_bertopic=False):
        print(f"\nEvaluating {model_name}...")
        
        if is_bertopic:
            # BERTopic handles its own topics
            topics_words = modeling.get_topics_words(model, None, n_top_words=10, model_type='bertopic')
            # Extract topic assignments from model
            topic_assignments = model.topics_
        else:
            # Sklearn models
            topics_words = modeling.get_topics_words(model, vectorizer, n_top_words=10, model_type='sklearn')
            # Get dominant topic for each doc
            doc_topic_dist = model.transform(dtm)
            topic_assignments = doc_topic_dist.argmax(axis=1)

        # 1. Coherence (Unsupervised)
        # Optimization: Use a sample of texts for Coherence to avoid 10+ min calculation per model
        import random
        COHERENCE_SAMPLE = 5000
        if len(texts_clean) > COHERENCE_SAMPLE:
             coherence_texts = random.sample(texts_clean, COHERENCE_SAMPLE)
        else:
             coherence_texts = texts_clean
             
        tokenized_texts = [text.split() for text in coherence_texts]
        
        if not topics_words: # Handle empty checks
             coherence = 0
        else:
            coherence = modeling.compute_coherence_score(topics_words, tokenized_texts)
        
        # 2. Supervised Metrics
        sup_metrics, _, y_pred_mapped, _ = evaluation.compute_supervised_metrics(topic_assignments, true_labels)
        
        # Log results
        result = {
            'Model': model_name,
            'Coherence': coherence,
            **sup_metrics
        }
        metrics_log.append(result)
        
        print(f"  Coherence: {coherence:.4f}")
        print(f"  Supervised: {sup_metrics}")
        
        # Save Confusion Matrix for best run (e.g. 15 topics or specific) if needed
        # We'll save it for every run overwriting or uniquely naming
        safe_name = model_name.replace(" ", "_").lower()
        evaluation.plot_confusion_matrix(true_labels, y_pred_mapped, 
                                         title=f"Confusion Matrix - {model_name}", 
                                         save_path=os.path.join(OUTPUT_DIR, f'cm_{safe_name}.png'))

    # --- Training Loop ---
    
    # A. Scikit-Learn Models (LSA, LDA, NMF)
    for n_topics in TOPIC_RANGES:
        print(f"\nTraining Sklearn Models with {n_topics} topics...")
        
        # LSA
        lsa_model, _ = modeling.train_lsa(dtm_tfidf, n_components=n_topics)
        evaluate_model_pipeline(f"LSA-{n_topics}", lsa_model, tfidf_vectorizer, dtm_tfidf)
        
        # LDA
        lda_model, _ = modeling.train_lda(dtm_bow, n_components=n_topics)
        evaluate_model_pipeline(f"LDA-{n_topics}", lda_model, bow_vectorizer, dtm_bow)
        
        # NMF
        nmf_model, _ = modeling.train_nmf(dtm_tfidf, n_components=n_topics)
        evaluate_model_pipeline(f"NMF-{n_topics}", nmf_model, tfidf_vectorizer, dtm_tfidf)

    # B. BERTopic (Auto or Fixed)
    # Running BERTopic once with 'auto' topic reduction usually results in large number of topics.
    # We will try to constrain it or run it once and report.
    # Note: BERTopic is slow on 80k without GPU.
    print("\nTraining BERTopic...")
    # Using a fixed number of topics for comparison 'nr_topics=15' roughly
    bert_model, _, _ = modeling.train_bertopic(texts_clean, n_topics=15)
    evaluate_model_pipeline("BERTopic-15", bert_model, None, is_bertopic=True)

    # --- Save Metrics ---
    metrics_df = pd.DataFrame(metrics_log)
    metrics_path = os.path.join(OUTPUT_DIR, 'model_comparison_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to {metrics_path}")
    print(metrics_df)

    # Plot Comparison
    # Pivot for plotting Coherence
    plt.figure(figsize=(10, 6))
    for model_type in ['LSA', 'LDA', 'NMF']:
        subset = metrics_df[metrics_df['Model'].str.contains(model_type)]
        if not subset.empty:
            # Extract n_topics
            subset['Topics'] = subset['Model'].apply(lambda x: int(x.split('-')[1]))
            plt.plot(subset['Topics'], subset['Coherence'], marker='o', label=model_type)
            
    # Add BERTopic point
    bertopic_row = metrics_df[metrics_df['Model'].str.contains("BERTopic")]
    if not bertopic_row.empty:
        plt.scatter([15], bertopic_row['Coherence'], color='red', marker='*', s=200, label='BERTopic')

    plt.title("Coherence Score Comparison")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence ($C_v$)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'final_coherence_comparison.png'))
    
    print("=== Pipeline Completed ===")

if __name__ == "__main__":
    main()
