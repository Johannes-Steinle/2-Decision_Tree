# Decision Tree Projekt

Meine Umsetzung der Decision Tree / Random Forest Übung aus dem Udemy-Kurs "Python für Data Science, Maschinelles Lernen & Visualization" im Rahmen der Angleichungsleistung.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Johannes-Steinle/2-Decision_Tree/main?filepath=Decision_Tree_Solution.ipynb)

## Überblick
Vorhersage, ob ein Kreditnehmer seinen Kredit vollständig zurückzahlt. Verglichen werden Decision Tree und Random Forest Klassifikatoren.

## Inhalt
* `Decision_Tree_Solution.ipynb` - Haupt-Notebook mit Datenanalyse, Modellvergleich und Evaluation
* `Loan_Data.csv` - Datensatz mit Kreditdaten

## Ausführung

1. Auf den **Binder-Badge** oben klicken, um das Notebook in myBinder zu starten.
2. Warten, bis die Umgebung geladen ist (kann 1-2 Minuten dauern).
3. `Decision_Tree_Solution.ipynb` öffnen.
4. Alle Zellen nacheinander ausführen (*Run > Run All Cells*).
5. **Erwartete Ergebnisse:**
   - Explorative Analyse der Kreditdaten mit Histogrammen und Countplots
   - Dummy-Variablen für die Spalte `purpose`
   - Training und Vergleich von Decision Tree und Random Forest
   - Classification Report — Random Forest schneidet besser ab
   - Accuracy von ca. **0.84 - 0.85** mit Random Forest

---

## Prüfungsaufgabe 2: Automatisierung und Testen

Ich habe das Projekt für Aufgabe 2 um Unit-Tests und Logging erweitert, nach dem Ansatz aus dem Artikel "Unit Testing and Logging for Data Science".

### Dateien
| Datei | Beschreibung |
|---|---|
| `model_logic.py` | Random Forest Logik mit `my_logger` und `my_timer` Dekoratoren |
| `test_model.py` | Unit-Tests für `predict()` (Accuracy) und `fit()` (Laufzeit) |
| `generate_test_data.py` | Skript zur Erzeugung der Testdaten |
| `train_data.csv` | Trainingsdaten (6704 Zeilen, mit Dummy-Variablen) |
| `test_data.csv` | Testdaten (2874 Zeilen) |
| `training.log` | Log-File mit Trainingsereignissen |

### Testfälle

**Testfall 1 - predict():** Das Random Forest Modell wird auf `train_data.csv` trainiert und die Accuracy auf `test_data.csv` geprüft. Ziel: Accuracy > 0.70.

**Testfall 2 - fit():** Die Laufzeit der Trainingsfunktion wird gemessen und geprüft, ob sie unter 120% der Normzeit (1.0s) bleibt.

### Testergebnisse
```text
[Test predict()] Gemessene Accuracy: 0.8441
.
[Test fit()] Gemessene Dauer: 0.7009s (Limit: 1.2000s)
.
----------------------------------------------------------------------
Ran 2 tests in 1.455s

OK
```

### Tests ausführen

1. Binder-Umgebung über den Badge oben starten.
2. **Terminal** öffnen (*File > New > Terminal*).
3. Folgenden Befehl ausführen:
   ```bash
   python -m unittest test_model -v
   ```
4. Die Tests laden die Daten aus `test_data.csv` und `train_data.csv`.
5. Beide Tests sollten mit `OK` durchlaufen.

Um die Testdaten neu zu generieren: `python generate_test_data.py`
