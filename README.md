# DEL - Mini Challenge 1
## Requirements 
Es liegt eine requirements.txt dabei, mit der die verwendeten Packages installiert werden können:

`pip install -r requirements.txt`

Es soll aber darauf geachtet werden, dass das Trainieren möglichst auf einer GPU durchgeführt werden kann, da eine Epoche ca. 15 Sekunde auf einer RTX 3060Ti brauchte. 

## Daten
Die Mini-Challenge benutzt die Daten aus der Challenge [3Da - Immobilienrechner](https://ds-spaces.technik.fhnw.ch/immobilienrechner/). 

Kurz, es wurde mithilfe von vielen gemeinde-spezifischen Daten Wohnobjekte klassifiziert (Handelt sich dieses Objekt um eine Wohnung? Haus?). Wir hatten diese Aufgabe in der Mini-Challenge mit Hilfe eines Boosted Gradient Trees gelöst. Dies kann man auch im Notebook `main.ipynb` nachschauen gehen. Da haben wir eine Macro-F1 Score von 0.46 erreicht.

## EDA
Eda wurde im Notebook `eda.ipynb` durchgeführt. 

## Preprocessing
Das Preprocessing kann in `helper_functions_preprocessing.py` gefunden werden. Darin findet man eine Python-Klasse, welche die gängisten Daten-Normalisierungen durchführt. Diese sind am Anfang vom Notebook `main.ipynb` genauer beschrieben.

## Eigentliche Mini-Challenge 
Die eigentliche Lösung der Mini-Challenge kann im Notebook `main.ipynb` gefunden werden. Sie wurde so aufgebaut, dass die möglichst der Aufgabenstellung definiert in `mini-challenges_SGDS_DEL_MC1.pdf` entspricht. Das Modell und wie es erstellt wurde kann in `main.ipynb` gefunden werden. 

### Grobstruktur NN
Es handelt sich um ein MLP mit mehreren Hidden Layers. Die Struktur des Netzes wurde auch durch Hyperparameter-Tuning gesucht. Pro Hidden Layer nimmt die Anzahl Neuronen mit folgender Formel ab:

`current_dim = int(current_dim / e**((i+1)/nr_layers))` 

`i` entspricht der Nummer des Hidden Layers (erstes Hidden Layer == 0, etc) und `nr_layers` der Anzahl der Hidden layers.

Somit wurde die Anzahl der hidden Layers und die Anfangsanzahl (vor dem Decay) durch Hyperparameter-Tuning gesucht.

### Early Stopping

Es hat "Early-Stopping" eingebaut. Das Trainieren wird unterbrochen, wenn sich die F1-Score für 10 Epochen auf dem Test-Set nicht verbessert.

## Hyperparameter-Tuning
Das Hyperparameter-Tuning wurde mit Hilfe von [Weights and Biases](https://wandb.ai/) durchgeführt und deren "Sweep"-Methode [Bayesian Hyperparameter Optimization](https://wandb.ai/site/articles/bayesian-hyperparameter-optimization-a-primer). 

Die gesamte Optimierung kann hier nachgeschaut werden: [vincenzo293/DEL-mini-challenge-1](https://wandb.ai/vincenzo293/DEL-mini-challenge-1?workspace=user-vincenzo293). Dies wird im Notebook `main.ipynb` aber genauer erklärt.

# Interessantes und Entdeckungen
## Schwieriger Anfang
Interessant war, dass ich am Anfang extrem schlecht unterwegs war. Ich hatte lange eine Macro-F1 Score von 0.21-0.23. Irgendwann hatte ich gemerkt, dass die Klasse nicht alle Werte korrekt standardisiert und es ein Attribut hatte, welcher Werte zwischen 30'000-120'000 annahm. Als ich dies in Ordnung gebracht habe (Standardisiert), stieg meine Macro-F1 Score auf 0.37-0.398.

## Optimizer Adam konnte nicht mit den Daten umgehen / ist falsch eingestellt
Mit SGD konnte ich das NN gut trainieren. Mit Adam ist meine F1-Score nie über 0.17 gekommen, selbst mit testen von vielen verschiedenen Learning-Rates (0.00001-0.001) und Hyperparametern. Dies wird man auch im Notebook `main.ipynb` genauer sehen.