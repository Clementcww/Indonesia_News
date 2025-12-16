
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

    # --- EDA PHASE 1: RAW DATA (Before Cleaning) ---
    print("\n[EDA] Generating RAW Data Visualizations...")
    # 1. Missing Values (Raw)
    eda.plot_missing_values_overall(df, title="Missing Values (Raw)", save_path=os.path.join(OUTPUT_DIR, 'missing_values_raw_overall.png'))
    eda.plot_missing_values_per_column(df, title="Missing Values per Column (Raw)", save_path=os.path.join(OUTPUT_DIR, 'missing_values_raw_column.png'))
    
    # 2. Category Distribution (Raw)
    eda.plot_category_distribution(df, column='tag1', save_path=os.path.join(OUTPUT_DIR, 'category_dist_raw.png'))
    
    # 3. Word Analysis (Raw)
    raw_content = df['Content'].astype(str).tolist()
    # Use a sample for raw wordcloud if heavy, but full is requested.
    eda.plot_top_words(raw_content, title="Top 20 Words (Raw)", save_path=os.path.join(OUTPUT_DIR, 'top_words_raw.png'))
    # Optional: Raw Wordcloud might be messy/slow but requested "all eda before after"
    eda.plot_wordcloud(raw_content, title="Word Cloud (Raw)", save_path=os.path.join(OUTPUT_DIR, 'wordcloud_raw.png'))


    # Check for duplicates/missing
    print(f"Original shape: {df.shape}")
    df.drop_duplicates(subset=['Content'], inplace=True)
    df.dropna(subset=['Content', 'tag1'], inplace=True) # Ensure tag1 is present for evaluation
    print(f"Shape after cleaning: {df.shape}")

    # 2. Preprocessing
    # Use Full Dataset now. If too slow, uncomment sample line.
    SAMPLE_SIZE = None 
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
    # Note: Using stem=False to speed up 80k rows.
    processed_path = os.path.join(OUTPUT_DIR, 'processed_full.csv')
    if os.path.exists(processed_path):
        print(f"[INFO] Loading cached preprocessed data from {processed_path}...")
        processed_df = pd.read_csv(processed_path)
        # Ensure clean_content is strictly string to avoid issues later
        processed_df['clean_content'] = processed_df['clean_content'].astype(str).fillna("")
    else:
        print("[INFO] Preprocessing (Stemming=False for efficiency on full dataset)...")
        processed_df = preprocessing.preprocess_dataframe(df, text_column='Content', sample_size=None, do_stem=False)
        # Save processed sample
        processed_df.to_csv(processed_path, index=False)
    
    texts_clean = processed_df['clean_content'].tolist()
    true_labels = processed_df['tag1'].tolist()
    
    # --- EDA PHASE 2: CLEAN DATA (After Preprocessing) ---
    print("\n[EDA] Generating CLEAN Data Visualizations...")
    
    # 1. Missing Values (Clean) - Should be empty/100% complete
    eda.plot_missing_values_overall(df, title="Missing Values (Clean)", save_path=os.path.join(OUTPUT_DIR, 'missing_values_clean_overall.png'))
    
    # 2. Category Distribution (Clean)
    eda.plot_category_distribution(df, column='tag1', save_path=os.path.join(OUTPUT_DIR, 'category_dist_clean.png'))
    
    # 3. Text Length Comparison
    eda.plot_text_length_distribution(processed_df, raw_col='Content', clean_col='clean_content', save_path=os.path.join(OUTPUT_DIR, 'text_length_comparison.png'))
    
    # 4. Word Analysis (Clean)
    eda.plot_top_words(texts_clean, title="Top 20 Words (Preprocessed)", save_path=os.path.join(OUTPUT_DIR, 'top_words_clean.png'))
    eda.plot_wordcloud(texts_clean, title="Word Cloud (Preprocessed)", save_path=os.path.join(OUTPUT_DIR, 'wordcloud_clean.png'))
    
    # 4. Modeling Prep
    print("\n--- Modeling Step ---")
    
    # Vectorization
    print("Vectorizing...")
    tfidf_vectorizer = modeling.get_vectorizer('tfidf', max_features=5000)
    dtm_tfidf = tfidf_vectorizer.fit_transform(texts_clean)
    
    bow_vectorizer = modeling.get_vectorizer('bow', max_features=5000)
    dtm_bow = bow_vectorizer.fit_transform(texts_clean)
    
    # Goal: Find best number of topics (3-15)
    TOPIC_RANGES = range(3, 16, 3) # 3, 6, 9, 12, 15
    # TOPIC_RANGES = [5] # Verification mode: single iteration

    
    metrics_log = []

    topic_summary_log = []

    def evaluate_model_pipeline(model_name, model, vectorizer, dtm=None, is_bertopic=False):
        print(f"\nEvaluating {model_name}...")
        
        topics_words = []
        if is_bertopic:
            # BERTopic handles its own topics
            topics_words = modeling.get_topics_words(model, None, n_top_words=10, model_type='bertopic')
        else:
            # Sklearn models
            topics_words = modeling.get_topics_words(model, vectorizer, n_top_words=10, model_type='sklearn')

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
        
        # Log results
        result = {
            'Model': model_name,
            'Coherence': coherence
        }
        metrics_log.append(result)
        
        # Log Topics for Qualitative Table
        topic_summary_log.append({
            'Model': model_name,
            'Topics': topics_words
        })
        
        print(f"  Coherence: {coherence:.4f}")

    # ==========================
    # PHASE 1: BASELINE MODELING
    # ==========================
    print("\n" + "="*40)
    print(" PHASE 1: BASELINE MODELING (n=5)")
    print("="*40)
    
    # LSA Baseline
    print("\nTraining Baseline LSA (n=5)...")
    lsa_base, _ = modeling.train_lsa(dtm_tfidf, n_components=5)
    evaluate_model_pipeline("Baseline-LSA", lsa_base, tfidf_vectorizer, dtm_tfidf)
    
    # LDA Baseline
    print("\nTraining Baseline LDA (n=5)...")
    lda_base, _ = modeling.train_lda(dtm_bow, n_components=5)
    evaluate_model_pipeline("Baseline-LDA", lda_base, bow_vectorizer, dtm_bow)
    
    # NMF Baseline
    print("\nTraining Baseline NMF (n=5)...")
    nmf_base, _ = modeling.train_nmf(dtm_tfidf, n_components=5)
    evaluate_model_pipeline("Baseline-NMF", nmf_base, tfidf_vectorizer, dtm_tfidf)
    
    # BERTopic Baseline
    print("\nTraining Baseline BERTopic (n=5)...")
    bert_base, _, _ = modeling.train_bertopic(texts_clean, n_topics=5)
    evaluate_model_pipeline("Baseline-BERTopic", bert_base, None, is_bertopic=True)
    
    # BASELINE SCORE CARD
    print("\n" + "-"*30)
    print(" BASELINE SCORE CARD")
    print("-"*30)
    for result in metrics_log:
        if "Baseline" in result['Model']:
            print(f"Model: {result['Model']:<20} | Coherence: {result['Coherence']:.4f}")
    print("-" * 30)

    # ==========================
    # PHASE 2: HYPERPARAMETER TUNING
    # ==========================
    print("\n" + "="*40)
    print(" PHASE 2: HYPERPARAMETER TUNING (Grid Search n=3-15)")
    print("="*40)
    
    TOPIC_RANGES = range(3, 16, 3) 
    # TOPIC_RANGES = [5] # Debug

    for n_topics in TOPIC_RANGES:
        print(f"\n>> Tuning Topic Count: {n_topics}...")
        
        # LSA
        lsa_model, _ = modeling.train_lsa(dtm_tfidf, n_components=n_topics)
        evaluate_model_pipeline(f"Tuned-LSA-{n_topics}", lsa_model, tfidf_vectorizer, dtm_tfidf)
        
        # LDA
        lda_model, _ = modeling.train_lda(dtm_bow, n_components=n_topics)
        evaluate_model_pipeline(f"Tuned-LDA-{n_topics}", lda_model, bow_vectorizer, dtm_bow)
        
        # NMF
        nmf_model, _ = modeling.train_nmf(dtm_tfidf, n_components=n_topics)
        evaluate_model_pipeline(f"Tuned-NMF-{n_topics}", nmf_model, tfidf_vectorizer, dtm_tfidf)
        
    
    # BERTopic (Fixed n=15)
    print("\nTraining Tuned BERTopic (Fixed n=15)...")
    bert_model, _, _ = modeling.train_bertopic(texts_clean, n_topics=15)
    evaluate_model_pipeline("Tuned-BERTopic-15", bert_model, None, is_bertopic=True)

    # ==========================
    # FINAL EVALUATION CARD
    # ==========================
    metrics_df = pd.DataFrame(metrics_log)
    metrics_path = os.path.join(OUTPUT_DIR, 'model_comparison_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    
    print("\n" + "="*40)
    print(" FINAL EVALUATION SCORE CARD (Best Configurations)")
    print("="*40)
    
    # Identify Best Model per Type
    for model_type in ['LSA', 'LDA', 'NMF']:
        subset = metrics_df[metrics_df['Model'].str.contains(model_type)]
        if not subset.empty:
            best_row = subset.loc[subset['Coherence'].idxmax()]
            print(f"  {model_type:<10}: {best_row['Model']:<20} (Coherence: {best_row['Coherence']:.4f})")
            
    # Save Qualitative Summary
    summary_path = os.path.join(OUTPUT_DIR, 'topic_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== QUALITATIVE TOPIC SUMMARY ===\n")
        for entry in topic_summary_log:
            f.write(f"\nModel: {entry['Model']}\n")
            for idx, topic in enumerate(entry['Topics']):
                f.write(f"  Topic {idx+1}: {', '.join(topic)}\n")
    print(f"\nTopic Summary saved to {summary_path}")

    # Plot Comparison (Tuned Models Only)
    plt.figure(figsize=(10, 6))
    for model_type in ['LSA', 'LDA', 'NMF']:
        # Filter for 'Tuned' models to plot the curve
        subset = metrics_df[metrics_df['Model'].str.contains(f"Tuned-{model_type}")]
        if not subset.empty:
            subset['Topics'] = subset['Model'].apply(lambda x: int(x.split('-')[-1]))
            subset = subset.sort_values('Topics')
            plt.plot(subset['Topics'], subset['Coherence'], marker='o', label=model_type)
            
    bertopic_row = metrics_df[metrics_df['Model'].str.contains("Tuned-BERTopic")]
    if not bertopic_row.empty:
        plt.scatter([15], bertopic_row['Coherence'], color='red', marker='*', s=200, label='BERTopic')
        
    plt.title("Coherence Score Comparison (Hyperparameter Tuning)")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence ($C_v$)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'final_coherence_comparison.png'))
    print("Saved tuning comparison plot.")

    print("=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
