import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging
import time
import os

# Logging Konfiguration
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

def load_data(filepath):
    """Lädt die Daten und führt Vorverarbeitung durch."""
    logger = logging.getLogger()
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Daten erfolgreich von {filepath} geladen.")
        
        # Vorverarbeitung (Dummy-Variablen für 'purpose')
        cat_feats = ['purpose']
        final_data = pd.get_dummies(df, columns=cat_feats, drop_first=True)
        
        X = final_data.drop('not.fully.paid', axis=1)
        y = final_data['not.fully.paid']
        return X, y
    except Exception as e:
        logger.error(f"Fehler beim Laden der Daten: {e}")
        raise

def fit_model(X_train, y_train):
    """Trainiert das RandomForest Modell und misst die Zeit."""
    logger = logging.getLogger()
    start_time = time.time()
    
    logger.info("Starte Modelltraining (Random Forest)...")
    model = RandomForestClassifier(n_estimators=100) # 100 ist schneller für Tests als 600
    model.fit(X_train, y_train)
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Training beendet in {duration:.4f} Sekunden.")
    
    return model, duration

def predict_model(model, X_test):
    """Erstellt Vorhersagen."""
    logger = logging.getLogger()
    logger.info("Erstelle Vorhersagen...")
    predictions = model.predict(X_test)
    return predictions

if __name__ == "__main__":
    X, y = load_data('Loan_Data.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    
    model, duration = fit_model(X_train, y_train)
    preds = predict_model(model, X_test)
    
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc}")
    logging.info(f"Modell-Accuracy im Testlauf: {acc}")
