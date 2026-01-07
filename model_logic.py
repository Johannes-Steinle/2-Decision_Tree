import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging
import time
import os
from functools import wraps

# Logging Konfiguration
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

def my_logger(orig_func):
    """Loggt den Funktionsnamen und die übergebenen Argumente."""
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info(f'Ran with args: {args}, and kwargs: {kwargs}')
        return orig_func(*args, **kwargs)
    return wrapper

def my_timer(orig_func):
    """Loggt die Ausführungszeit der Funktion."""
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        logging.info(f'{orig_func.__name__} ran in: {t2:.4f} sec')
        return result
    return wrapper

@my_logger
@my_timer
def load_data(filepath):
    """Lädt die Daten und führt Vorverarbeitung durch."""
    try:
        df = pd.read_csv(filepath)
        # Vorverarbeitung (Dummy-Variablen für 'purpose')
        cat_feats = ['purpose']
        final_data = pd.get_dummies(df, columns=cat_feats, drop_first=True)
        
        X = final_data.drop('not.fully.paid', axis=1)
        y = final_data['not.fully.paid']
        return X, y
    except Exception as e:
        logging.error(f"Fehler beim Laden der Daten: {e}")
        raise

@my_logger
@my_timer
def fit_model(X_train, y_train):
    """Trainiert das RandomForest Modell."""
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

@my_logger
@my_timer
def predict_model(model, X_test):
    """Erstellt Vorhersagen."""
    predictions = model.predict(X_test)
    return predictions

if __name__ == "__main__":
    X, y = load_data('Loan_Data.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    
    model = fit_model(X_train, y_train)
    preds = predict_model(model, X_test)
    
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc}")
