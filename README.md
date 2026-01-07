# Decision Tree Projekt

Dieses Repository enthält ein Decision Tree und Random Forest Projekt als Teil der Angleichungsleistungen im Modul "Data Science und Engineering mit Python".

## Projektüberblick
Das Ziel dieses Projekts ist es, vorherzusagen, ob ein Kreditnehmer seinen Kredit vollständig zurückzahlen wird.

## Inhalt
* `Decision_Tree_Solution.ipynb`: Das Haupt-Notebook mit der Analyse und dem Modell.
* `Loan_Data.csv`: Der Datensatz, der für Training und Test verwendet wurde.

## Prüfungsaufgabe 2: Automatisierung und Testen

Dieses Projekt wurde gemäß den Anforderungen für Aufgabe 2 refaktoriert und mit automatisierten Tests sowie Logging ausgestattet.

### Struktur
- `model_logic.py`: Enthält die Kernlogik (Random Forest) sowie Logging-Funktionalität.
- `test_model.py`: Führt Unit-Tests zur Validierung der Modellgüte (Accuracy) und der Trainingslaufzeit durch.
- `training.log`: Protokolliert Trainingsereignisse.

### Testergebnisse
Die Tests wurden erfolgreich ausgeführt:
```text
[Test predict()] Gemessene Accuracy: 0.8445
.
[Test fit()] Gemessene Dauer: 0.6945s (Limit: 1.5000s)
.
----------------------------------------------------------------------
Ran 2 tests in 1.440s

OK
```

## Nutzung
Das Notebook kann direkt über [myBinder](https://mybinder.org/v2/gh/Johannes-Steinle/2-Decision_Tree/main?filepath=Decision_Tree_Solution.ipynb) ausgeführt werden.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Johannes-Steinle/2-Decision_Tree/main?filepath=Decision_Tree_Solution.ipynb)
