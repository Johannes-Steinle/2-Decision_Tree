import unittest
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from model_logic import load_data, fit_model, predict_model

class TestMLModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Wird einmal vor allen Tests ausgeführt. Lädt Daten und bereitet sie vor."""
        cls.X, cls.y = load_data('Loan_Data.csv')
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.3, random_state=101
        )
        # Norm-Zeit für Random Forest Training ca. 1.0s (Anpassung je nach CPU)
        cls.norm_fit_time = 2.0 

    def test_1_predict_accuracy(self):
        """
        Ziel: Accuracy muss > 0.70 sein (Random Forest auf Loan Data).
        """
        model, _ = fit_model(self.X_train, self.y_train)
        predictions = predict_model(model, self.X_test)
        acc = accuracy_score(self.y_test, predictions)
        
        print(f"\n[Test Predict] Gemessene Accuracy: {acc}")
        self.assertGreater(acc, 0.70, f"Accuracy zu niedrig: {acc} < 0.70")

    def test_2_fit_runtime(self):
        """
        Ziel: Laufzeit < 120% der Normzeit.
        """
        _, duration = fit_model(self.X_train, self.y_train)
        
        limit = self.norm_fit_time * 1.2
        print(f"\n[Test Fit] Gemessene Dauer: {duration:.4f}s (Limit: {limit:.4f}s)")
        
        self.assertLess(duration, limit, f"Training dauerte zu lange: {duration:.4f}s > {limit:.4f}s")

if __name__ == '__main__':
    unittest.main()
