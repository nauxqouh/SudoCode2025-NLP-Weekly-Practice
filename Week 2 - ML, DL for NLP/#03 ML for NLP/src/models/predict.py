import joblib
from src.preprocessing.cleaner import preprocessing
from src.preprocessing.vectorizer import get_vectorizer
from config import MODEL_PATH

def load_model(model_path=MODEL_PATH):
    """
    Load trained model
    """
    
    model = joblib.load(model_path)
    return model

def predict_text(text, model, vectorizer):
    """
    Predict new data
    """

    text_clean = preprocessing(text)
    X_vec = vectorizer.transform([text_clean])
    y_pred = model.predict(X_vec)
    
    return y_pred[0]
