from data.datasets import load_data, split_data
from src.preprocessing.cleaner import preprocessing
from src.preprocessing.vectorizer import get_vectorizer
from src.models.train import train_model
from src.models.evaluate import evaluate
from src.models.predict import load_model, predict_text

def main():
    # Load data
    df = load_data()
    df["text"] = df["text"].apply(preprocessing)
    
    # Vectorize
    vectorizer = get_vectorizer("tfidf")
    X_train_vec = vectorizer.fit_transform(df["text"])
    y_train = df["label"]
    
    # Train
    model = train_model(X_train_vec, y_train, model_type="svm")
    
    # Predict
    sample_text = "Sản phẩm này rất tuyệt vời ❤"
    y_pred = predict_text(sample_text, model, vectorizer)
    print(f"Văn bản: {sample_text}")
    print(f"Dự đoán: {y_pred}")

if __name__ == "__main__":
    main()
