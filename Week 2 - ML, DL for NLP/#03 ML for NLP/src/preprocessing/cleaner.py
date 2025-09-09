import unicodedata
import re
import emoji
from underthesea import word_tokenize
from config import EMOJI_MAP, CUSTOM_STOPWORDS

def replace_emoji_words(text, emoji_map):
    for k, v in emoji_map.items():
        text = text.replace(k, v.replace(" ", " "))
    return text

def preprocessing(text):
    # 1. Cleaning
    ## UNICODE normalization
    text = unicodedata.normalize("NFC", str(text))
    ## lowercase
    text = text.lower()
    ## remove HTML tags
    text = re.sub(r'<.*?>', '', text) 
    ## remove URLs like http/www & domains like .com, .vn, .net, ...
    text = re.sub(r'http\S+|www\S+', ' ', text) 
    text = re.sub(r'\b[\w-]+\.(?:com|vn|net|org|info|gov|edu)(?:\.[a-z]{2})?\b', ' ', text)
    ## keep emoji and replace by text
    text = emoji.demojize(text)
    text = replace_emoji_words(text, EMOJI_MAP)
    ## remove punctuation, strange characters
    text = re.sub(r'[^\w\s]', ' ', text)
    ## remove extra white space
    text = re.sub(r"\s+", " ", text).strip()

    # 2. Tokenization
    tokens = word_tokenize(text, format="list")
    
    # 3. Remove stopwords
    tokens = [w for w in tokens if w not in CUSTOM_STOPWORDS]
    
    return " ".join(tokens)