# Changelog

## 0.6.3

- erster vertikaler Prototyp für strukturierte LLM-zu-Adventure-Erzeugung
- provider-neutraler OpenAI-kompatibler Director mit explizitem Responses-v1- oder Chat-Completions-JSON-Schema-Protokoll ohne Fallback
- versioniertes providerfreundliches WireSpec mit deterministischer AdventureSpec-Transformation und Schema-Preflight
- robuster HTTP-Transport mit getrenntem Connect-/Read- und hartem Total-Timeout
- explizite begrenzte LLM-Laufzeitoptionen über CLI und Umgebung mit persistierter effektiver Provenienz
- begrenzter einmaliger Repair-Loop und atomare Director-Provenienz
- injizierbarer, timeout- und größenbegrenzter HTTP-Transport ohne Redirect-Folgen
- strikt validierte AdventureSpec, RoomSpec und Scene Graphs
- versionierte Theme-Ontologie mit fünf Themes
- deterministische 128×72-Golden-Room-Kompilierung
- Vektor-Walkboxes, Kollisionsprüfung und geglättete Pfadsuche
- deklarative Headless-Runtime mit validiertem Save/Load
- begrenzter State-Space-Solver und vollständiger Vorabvalidator
- offline nutzbarer HTML/Canvas-Browserexport mit eigenen Platzhaltern
- CLI für Generierung, Validierung und Lösung

## 0.6.1

- absolute X/Y-Lernziele durch Terrainregion und einen von 16 Anchors ersetzt
- Regionen für Strand, offenes Land, Felsfeld und Wald
- Landmark Slots für Kiste, NPC, Portal und Ruine
- deterministischer Vegetations-Scatter-Layer mit Mindestabstand
- Waldstufe und Vegetationsdichte als zusätzliche Terrainparameter
- eigener Placement Encoder für Region und Anchor
- 14.000 Trainingslandschaften und 45 Epochen
- 1.000 Generatorwelten und 200 pixelgenaue Landmark-Round-trips validiert
- Anchor-Suche durch vektorisierte Integralbilder beschleunigt
- 14.000 Trainingslandschaften werden einmalig vorberechnet statt in jeder Epoche neu erzeugt
- CUDA-Gerät, GPU-Name, gepinnter Speicher und asynchrone Batch-Transfers ergänzt

## 0.6 – Meilenstein 3

- Terrain Graph mit Biom, Küstenrichtung, Uferlinie, Strandbreite und Felsigkeit
- Terrainklassen Wasser, Sand, Gras, Erde, Fels und Schnee
- deterministischer Rasterizer für organische Küstenlinien
- Walkability-, Terrain-, Object- und Interaction-Maps
- Außenwelt-Slots für Baum, Fels, NPC und Portal
- getrennte Terrain-, Geometry-, Presence- und Attribute Encoder
- 12.000 Trainingslandschaften und 45 Epochen
- 5.000 Generatorwelten ohne Objektüberlappung oder Wasserplatzierung validiert
- vollständigen 45-Epochen-Referenzlauf, Endmetriken und Visualisierung archiviert

## 0.5.2

- eigener Attribute Encoder für Klasse, Aktion und Trigger
- eigener Attribute Slot Decoder mit gemeinsam genutzten Slot Queries
- Geometry-, Presence- und Transition-Pfade gegenüber 0.5.1 unverändert
- kontrollierter A/B-Test gegen die geteilte Slot-Repräsentation aus 0.5.1

## 0.5.1

- Seed-Token-Target, 256-Klassen-Kopf, Loss und MAE-Metrik entfernt
- Folgewelt-ID deterministisch aus Welt-Seed, Slot-ID, Trigger-Typ und Story-State abgeleitet
- Slot-Latent von sieben auf sechs tatsächlich gelernte Merkmale reduziert
- 0.5 bleibt als unveränderter Vergleichsstand erhalten

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
