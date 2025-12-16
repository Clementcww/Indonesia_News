# Indonesia News Topic Modeling

## 📌 Project Overview
This project implements an **Unsupervised Topic Modeling** pipeline to extract latent themes from a dataset of **80,000+ Indonesian News Articles**. 

The goal was to categorize unstructured news text into coherent topics (e.g., Politics, Economy, Crime, Sports) without using pre-existing labels, comparing multiple algorithms to find the best approach.

## 🛠️ Tech Stack
-   **Language**: Python 3.10+
-   **Libraries**:
    -   `pandas`, `numpy` (Data Manipulation)
    -   `scikit-learn` (LSA, NMF, LDA, Vectorization)
    -   `gensim` (Coherence Metrics)
    -   `bertopic` (Deep Learning Topic Modeling)
    -   `matplotlib`, `seaborn`, `wordcloud` (Visualization)
    -   `Sastrawi` (Indonesian NLP Preprocessing)

## 📊 Pipeline Workflow

### 1. Data Preprocessing
-   **Cleaning**: Removal of duplicates and null values.
-   **Normalization**: Lowercasing, punctuation removal.
-   **Stopword Removal**: Using `Sastrawi` to remove common Indonesian stopwords (e.g., "dan", "yang", "di").
-   **Stemming**: Reducing words to their root forms (e.g., "pembangunan" -> "bangun") to consolidate topic terms.

### 2. Exploratory Data Analysis (EDA)
Comprehensive analysis performed **Before** and **After** preprocessing:
-   **Missing Values Analysis**: Global Pie Chart & Column-wise Bar Chart.
-   **Word Frequency**: Top 20 words bar plots.
-   **Word Clouds**: Visualizing corpus density.
-   **Distribution**: Article category and text length distributions.

### 3. Modeling Strategy
We implemented and compared four distinct algorithms:
1.  **LSA (Latent Semantic Analysis)**: Dimensionality reduction using SVD on TF-IDF.
2.  **LDA (Latent Dirichlet Allocation)**: Probabilistic generative model on BoW.
3.  **NMF (Non-negative Matrix Factorization)**: Matrix factorization enforcing non-negativity (often better for interpretability).
4.  **BERTopic**: Transformer-based embeddings (BERT) with c-TF-IDF.

**Evaluation Protocol**:
-   **Phase 1: Baseline**: Run all models with `n_topics=5`.
-   **Phase 2: Hyperparameter Tuning**: Grid search for `n_topics` in range `[3, 6, 9, 12, 15]`.
-   **Metric**: **Coherence Score ($C_v$)** (Higher is better).

## 🏆 Key Results

After extensive tuning, **NMF (Non-negative Matrix Factorization)** emerged as the best model.

| Model | Configuration | Coherence Score ($C_v$) | Key Characteristics |
| :--- | :--- | :--- | :--- |
| **NMF** | **15 Topics** | **0.8307** | **Best Interpretability.** Distinct, non-overlapping topics. |
| **BERTopic** | Auto (n=15) | 0.7314 | Strong baseline, good at capturing specific nuances. |
| **LSA** | 3 Topics | 0.7216 | Good for broad themes but broad overlaps at higher K. |
| **LDA** | 12 Topics | 0.6107 | Struggled with noisy/overlapping terms in this dataset. |

### Discovered Topics (NMF-15)
The model successfully identified 15 distinct themes without supervision:
1.  **General News**: Footer text/ads (`berita`, `klik`, `baca`).
2.  **Crime**: Police cases (`korban`, `pelaku`, `polisi`).
3.  **Political Parties**: Internal politics (`partai`, `golkar`, `pdi`).
4.  **Corruption**: KPK cases (`kpk`, `korupsi`, `tersangka`).
5.  **National Leadership**: President & Cabinet (`prabowo`, `presiden`, `menteri`).
6.  **Sports**: Football match reports (`pemain`, `laga`, `gol`).
7.  **Urban Issues**: Fires, floods, traffic (`jalan`, `kebakaran`, `banjir`).
8.  **International Conflict**: Middle East updates (`israel`, `palestina`, `gaza`).
9.  **Military**: TNI/Defense (`tni`, `prajurit`, `panglima`).
10. **Social Programs**: School meals/Education (`sekolah`, `anak`, `makan bergizi`).
11. **Hajj/Umrah**: Religious pilgrimage (`haji`, `jemaah`, `kuota`).
12. **Regional Elections (Pilkada)**: Jakarta Governor race (`pramono`, `ridwan kamil`).
13. **Religion**: Ramadan activities (`puasa`, `ramadhan`).
14. **Law**: Legislation & Court (`undang`, `mk`, `pasal`).
15. **Economy**: Budget & Finance (`rp`, `triliun`, `anggaran`).

## 📁 Repository Structure
```
├── src/
│   ├── preprocessing.py  # Text cleaning & normalization
│   ├── eda.py           # Visualization functions
│   ├── modeling.py      # Model training (LSA, LDA, NMF, BERTopic)
│   ├── evaluation.py    # Metric calculation
│   └── visualize_results.py # Result plotting script
├── output/              # Generated plots and metrics CSV
├── main.py              # Main execution pipeline
├── final_merge_dataset.csv # (Source Data)
└── README.md            # Project Documentation
```

## 🚀 How to Run
1.  **Install Dependencies**:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn wordcloud gensim bertopic Sastrawi
    ```
2.  **Run the Pipeline**:
    ```bash
    python main.py
    ```
    *This will load the data, run EDA, train all models, and save results to the `output/` directory.*