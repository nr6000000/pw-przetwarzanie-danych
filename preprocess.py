import re
from collections import Counter
from html import unescape

import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


def extract_words(texts: list[str]):
    words = [re.sub(r'[^A-Za-z\s]', ' ', text).lower().split() for text in texts]
    return words

"""
Strips HTML tags (doesn't lower letters and leaves special characters) 
"""
def strip_html(texts: list[str]):
    def strip_one(text: str):
        html_text = unescape(text)
        return BeautifulSoup(html_text, features='html.parser').get_text()

    return [strip_one(text) for text in texts]

"""
Extracts from text proper words only and removes ones not included 
in provided dictionary
"""
def strip_dict(texts: list[str], dictionary: set[str], exclude=False):
    words = extract_words(texts)
    def strip_one(words: list[str]):
        return ' '.join(word for word in words
                        if (word in dictionary) != exclude)
    
    return [strip_one(text_words) for text_words in words] 

"""
Extracts from text proper words only and removes stopwords
"""
def strip_stopwords(texts: list[str]):
    nltk.download('stopwords')
    return strip_dict(texts, set(stopwords.words('english')), exclude=True)

"""
Extracts from text proper words only and removes most and least frequent
"""
def strip_by_freq(texts: list[str], min_df: float, max_df: float):
    words = extract_words(texts)
    
    # Maximum one word per text
    words_exists = [list(set(text_words)) for text_words in words]
    words_flat = [words for words in words_exists for words in words]
    words_count = Counter(words_flat)
    allowed_words = {word for word, count in words_count.items()
                        if min_df < count/len(texts) < max_df}
    
    return strip_dict(texts, allowed_words)
