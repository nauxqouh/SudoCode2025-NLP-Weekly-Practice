from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate(y_pred, y_test, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    classification_rp = classification_report(y_test, y_pred)
    cfs_mx = confusion_matrix(y_test, y_pred)
    main_metric = accuracy
    return main_metric