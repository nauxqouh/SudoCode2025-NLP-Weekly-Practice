import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from config import MODEL_PATH

def train_model(X_train_vec, y_train, model_type="svm"):
    if model_type == "nb":
        model = MultinomialNB()
    elif model_type == "svm":
        model = SVC(kernel='rbf', class_weight='balanced')
    else:
        raise ValueError("Model type must be 'nb' or 'svm'")
    
    model.fit(X_train_vec, y_train)

    return model
