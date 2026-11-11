import re
import warnings
from collections import Counter
from html import unescape
from pathlib import Path

import nltk
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

TOKEN_PATTERN = re.compile(r'(?u)\b\w\w+\b')


def extract_words(texts: list[str]):
    words = [re.sub(r'[^A-Za-z\s]', ' ', text).lower().split() for text in texts]
    return words

"""
Strips HTML tags (doesn't lower letters and leaves special characters) 
"""
class HtmlCleaner(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        def strip_one(text: str):
            html_text = unescape(text)
            return BeautifulSoup(html_text, features='html.parser').get_text(separator=' ')

        return [strip_one(text) for text in X]    

"""
Extracts from text proper words only and removes ones not included 
in provided dictionary
"""
class DictionaryCleaner(TransformerMixin, BaseEstimator):
    def __init__(
            self, 
            dictionary: set[str] | None = None, 
            dictionary_path: str | None = None, 
            exclude=False):
        self.dictionary_path = dictionary_path
        if dictionary is not None:
            self.dictionary = dictionary
        elif dictionary_path is not None:
            with open(dictionary_path, "r", encoding="utf-8") as file:
                self.dictionary = {line.strip() for line in file if line.strip()}
        else:
            raise ValueError("Either pass dictionary or dictionary_path")

        self.exclude = exclude

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        def strip_one(match):
            word = match.group()
            return word if (word.lower() in self.dictionary) != self.exclude else ' '
        
        return [TOKEN_PATTERN.sub(strip_one, text) for text in X]

"""
Extracts from text proper words only and removes stopwords
"""
class StopwordCleaner(DictionaryCleaner):
    def __init__(self):
        nltk.download('stopwords', quiet=True)
        super().__init__(set(stopwords.words('english')), exclude=True)

"""
Extracts from text proper words only and removes most and least frequent
"""
class FreqWordCleaner(TransformerMixin, BaseEstimator):
    def __init__(self, min_df: float, max_df: float):
        self.min_df = min_df
        self.max_df = max_df

    def fit(self, X, y=None):
        words = extract_words(X)
    
        # Maximum one word per text
        words_exists = [list(set(text_words)) for text_words in words]
        words_flat = [word for text in words_exists for word in text]
        words_count = Counter(words_flat)
        self.allowed_words = {word for word, count in words_count.items()
                            if self.min_df <= count/len(words) <= self.max_df}

        return self
    
    def transform(self, X):
        words = extract_words(X)
        return [' '.join(word for word in text_words if word in self.allowed_words)
                for text_words in words]
