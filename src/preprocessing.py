
import re
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def get_stopword_remover():
    factory = StopWordRemoverFactory()
    # You can add more custom stopwords here if needed
    # more_stopwords = [...]
    # factory.get_stop_words() + more_stopwords
    return factory.create_stop_word_remover()

def get_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

def normalize_text(text):
    """
    Normalizes text:
    1. Lowercase
    2. Remove numbers and special characters
    3. Remove extra whitespace
    Reason: Standardization ensures that 'Kata' and 'kata' are treated as the same, 
    and punctuation/numbers often don't carry topical meaning in this context.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text) # Keep only alphabets and spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_packet(text, stopword_remover, stemmer, do_stem=True):
    """
    Applies the full preprocessing pipeline on a single string.
    """
    # 1. Normalization
    text = normalize_text(text)
    
    # 2. Stopword Removal
    # Reason: Remove common words (di, ke, dari) that appear frequently but have little semantic value for topic modeling.
    text = stopword_remover.remove(text)
    
    # 3. Stemming (Optional but recommended)
    # Reason: Reduce words to their root form (e.g., "memakan" -> "makan") to reduce feature space and group related terms.
    # Note: Sastrawi stemming is computationally expensive.
    if do_stem:
        text = stemmer.stem(text)
        
    return text

def preprocess_dataframe(df, text_column='Content', sample_size=None, do_stem=True):
    """
    Orchestrates the preprocessing on the dataframe.
    """
    if sample_size:
        print(f"Sampling {sample_size} rows for processing speed...")
        df = df.sample(n=sample_size, random_state=42).copy()
    else:
        df = df.copy()

    print("Initializing Sastrawi tools...")
    stopword_remover = get_stopword_remover()
    stemmer = get_stemmer()

    print(f"Starting preprocessing on {len(df)} rows. Stemming={do_stem}...")
    
    # Apply preprocessing
    # Using a simple apply might be slow for stemming. 
    # For large datasets, consider parallelization or progress bars (tqdm) if interactive.
    df['clean_content'] = df[text_column].apply(
        lambda x: preprocess_packet(x, stopword_remover, stemmer, do_stem)
    )
    
    print("Preprocessing complete.")
    return df
