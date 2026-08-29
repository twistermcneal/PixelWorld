# Changelog

## 0.5 – Meilenstein 2

- variable Anzahl von ein bis acht Object Slots
- frei vorhergesagte Objektklasse je Slot
- Aktionen `LOOK`, `USE` und `SCAN`
- Trigger-Typen `NONE`, `WORLD`, `STORY` und `SECRET`
- Seed-Token von 0 bis 255 für deterministische Folgewelten
- Multi-Task-Training und erweiterte Auswertung
- Referenzlauf: Presence 0,981, relative Position 0,495 px, Interaction IoU 0,686
- offener Fehler: Seed-Token MAE 65,1 und kein exakter Treffer

## 0.4.4 – Meilenstein 1

- Presence Encoder vom Geometry Encoder getrennt
- Presence Accuracy im Referenzlauf auf 0,975 verbessert
- relatives Positions-MAE von 0,419 Pixeln
- Interaction IoU von 0,807

## 0.4.3

- Objektpositionen relativ zum Raumursprung
- getrennte Messung relativer und absoluter Positionsfehler
- feste, klassenspezifische Objektgrößen

## 0.4.2

- ordinaler Koordinaten-Loss mit gaußförmigen Soft Targets
- Dekodierung über den erwarteten Pixelwert
- getrennte Positions- und Größenmetriken

## 0.4.1

- Koordinatenklassifikation auf ganzzahlige Pixelklassen von 0 bis 64
- pixelgenauer Scene-Graph-Rasterizer

## 0.4

- erster Scene-Graph-basierter Generator mit festen Rollen für Tür, NPC, Objekt und Portal
