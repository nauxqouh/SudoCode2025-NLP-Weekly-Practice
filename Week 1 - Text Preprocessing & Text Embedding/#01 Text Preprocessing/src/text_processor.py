import re
import json
import csv
import time
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
    
        text = re.sub(r'(?:email:|e-mail:)?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', ' ', text) # Remove emails
        
        # Remove URLs like http/www & domains like .com, .vn, .net, ...
        text = re.sub(r'http\S+|www\S+', ' ', text) 
        text = re.sub(r'\b[\w-]+\.(?:com|vn|net|org|info|gov|edu)(?:\.[a-z]{2})?\b', ' ', text)
        
        # Remove copyrighter
        text = re.sub(r'giấy phép.*$', ' ', text)
        text = re.sub(r'chính sách.*$', ' ', text)
        text = re.sub(r'địa chỉ.*$', ' ', text)
        text = re.sub(r'rss$', ' ', text)
        
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
        
        text = self.normalize(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        
        if self.vectorizer_flag:
            return ' '.join(tokens)
        else:
            return tokens
        

class NewsProcessor():
    def __init__(self, stopwords_path=None, vectorizer_flag=True):
        self.stopwords = set()
        if stopwords_path:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                self.stopwords = set([line.strip() for line in f if line.strip()])
        self.processor = TextProcessor(stopwords=self.stopwords, vectorizer_flag=vectorizer_flag)
        self.vectorizer_flag = vectorizer_flag
    
    def process(self, dat_news, early_stopping=None):
        """
        Processing all documents in news dataset
        """
        start_time = time.time()
        processed_contents = []
        for i, doc in enumerate(dat_news):
            content = doc['content']
            processed_content = self.processor.process(content)
            
            processed_contents.append({'id': doc['id'], 'content': processed_content})
            
            # Showing process for 30 first documents
            if i < 30:
                print(f"Document {i+1}:")
                print(f"Original content: {content}\n")
                print(f"Processed content: {processed_content}\n")
            
            if (i+1) % 1000 == 0:
                print("-"*110)
                print(f"\nProcessed {i+1} documents. Time: {time.time() - start_time}.\n")
                print("-"*110)
                start_time = time.time()
            
            # dừng sớm nếu early_stopping được set
            if early_stopping and (i+1) >= early_stopping:
                break
            
        return processed_contents
    
    def save(self, processed_news, output_path):
        """Save the processed text results to file
        *.txt: vectorizer purpose
        *.json: embedding purpose
        """
        if self.vectorizer_flag:
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['id', 'content'])
                writer.writeheader()
                for doc in processed_news:
                    writer.writerow({'id': doc['id'], 'content': doc['content']})
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_news, f, ensure_ascii=False, indent=2)