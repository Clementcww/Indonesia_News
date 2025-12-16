
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation, NMF
from sklearn.model_selection import train_test_split
import gensim
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from bertopic import BERTopic

def get_vectorizer(type='tfidf', max_features=5000):
    """
    Returns the vectorizer.
    Using 'max_features' to limit vocabulary size for performance and noise reduction.
    """
    if type == 'tfidf':
        # Used for LSA usually, but can be used for LDA (though counts are better for LDA probabilistic model)
        return TfidfVectorizer(max_features=max_features, use_idf=True)
    elif type == 'bow':
        # Bag of Words - counts. Good for LDA.
        return CountVectorizer(max_features=max_features)
    else:
        raise ValueError("Type must be 'tfidf' or 'bow'")

def train_lsa(dtm, n_components=5):
    """
    Trains LSA (TruncatedSVD).
    """
    lsa_model = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_topic_matrix = lsa_model.fit_transform(dtm)
    return lsa_model, lsa_topic_matrix

def train_lda(dtm, n_components=5):
    """
    Trains LDA.
    """
    lda_model = LatentDirichletAllocation(n_components=n_components, random_state=42, n_jobs=None)
    lda_topic_matrix = lda_model.fit_transform(dtm)
    return lda_model, lda_topic_matrix

def train_nmf(dtm, n_components=5):
    """
    Trains NMF.
    """
    nmf_model = NMF(n_components=n_components, random_state=42, init='nndsvd')
    nmf_topic_matrix = nmf_model.fit_transform(dtm)
    return nmf_model, nmf_topic_matrix

def train_bertopic(texts, n_topics='auto'):
    """
    Trains BERTopic model.
    n_topics: 'auto' or int.
    """
    # Using small model 'all-MiniLM-L6-v2' for efficiency
    # If n_topics is int, bertopic uses nr_topics arg
    topic_model = BERTopic(embedding_model='sentence-transformers/all-MiniLM-L6-v2', 
                           nr_topics=n_topics, 
                           verbose=True)
    topics, probs = topic_model.fit_transform(texts)
    return topic_model, topics, probs

def get_topics_words(model, vectorizer, n_top_words=10, model_type='sklearn'):
    """
    Extracts top words for each topic.
    model_type: 'sklearn' (LSA/LDA/NMF) or 'bertopic'
    """
    if model_type == 'bertopic':
        # BERTopic API is different
        topic_info = model.get_topic_info()
        topics_list = []
        # Filter out outlier topic -1 if present, or handle it
        # BERTopic topics are 0-indexed (or -1 for noise)
        unique_topics = sorted([t for t in topic_info['Topic'] if t != -1])
        
        for topic_id in unique_topics:
            words = [word for word, _ in model.get_topic(topic_id)][:n_top_words]
            topics_list.append(words)
        return topics_list

    # Sklearn models
    feature_names = np.array(vectorizer.get_feature_names_out())
    topics = []
    
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = feature_names[top_features_ind]
        topics.append(top_features.tolist())
    
    return topics

def compute_coherence_score(topics, texts, dictionary=None, coherence='c_v'):
    """
    Computes Coherence Score using Gensim.
    texts: List of tokenized texts (list of lists of strings).
    """
    if dictionary is None:
        dictionary = Dictionary(texts)
    
    coherence_model = CoherenceModel(
        topics=topics, 
        texts=texts, 
        dictionary=dictionary, 
        coherence=coherence
    )
    return coherence_model.get_coherence()

def compute_coherence_per_topic(topics, texts, dictionary=None, coherence='c_v'):
    """
    Returns coherence score for EACH topic.
    """
    if dictionary is None:
        dictionary = Dictionary(texts)
    
    coherence_model = CoherenceModel(
        topics=topics, 
        texts=texts, 
        dictionary=dictionary, 
        coherence=coherence
    )
    # get_coherence_per_topic() returns list of scores
    return coherence_model.get_coherence_per_topic()

