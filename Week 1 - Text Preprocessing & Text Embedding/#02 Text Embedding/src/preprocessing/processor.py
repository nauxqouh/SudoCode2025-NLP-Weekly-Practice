import re
from underthesea import word_tokenize

class TextProcessor:
    def __init__(self, stopwords=None, vectorizer_flag=True):
        self.stopwords = stopwords if stopwords else set()
        self.vectorizer_flag = vectorizer_flag
        
    def normalize(self, text):
        """
        Normalize and cleaning raw text
        """
        text = text.lower() # Lowercase

        text = re.sub(r'<.*?>', '', text) # Remove HTML tags
        
        # Remove URLs like http/www & domains like .com, .vn, .net, ...
        text = re.sub(r'http\S+|www\S+', ' ', text) 
        text = re.sub(r'\b[\w-]+\.(?:com|vn|net|org|info|gov|edu)(?:\.[a-z]{2})?\b', ' ', text)
        
        text = re.sub(r'\d+', ' ', text) # Remove digits
        text = re.sub(r'[^\w\s]', ' ', text) # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespaces

        return text
    
    def tokenize(self, text):
        """Tokenization using underthesea library"""
        return word_tokenize(text, format="list")
    
    def remove_stopwords(self, tokens):
        """Stop words removal"""
        return [word for word in tokens if word not in self.stopwords]

    def process(self, text):
        """
        Process completely one document.
        1. Lowercase and Cleaning text using RegEx
        2. Tokenization using underthesea library
        3. Remove stopwords
        4. Treat vectorizer flag
        """

        print("Loading normalization...")
        text = self.normalize(text)
        print("Loading tokenization...")
        tokens = self.tokenize(text)
        print("Loading stopwords removal...")
        tokens = self.remove_stopwords(tokens)
        
        if self.vectorizer_flag:
            return ' '.join(tokens)
        else:
            return tokens