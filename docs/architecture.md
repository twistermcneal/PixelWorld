# Architektur von PixelWorld 0.6

## Ziel

PixelWorld trennt die Erzeugung einer Welt in zwei Ebenen:

1. Ein lernendes Modell erzeugt einen Terrain- und Object Scene Graph.
2. Ein deterministischer Rasterizer übersetzt diesen Graphen in Pixel- und Logik-Maps.

So bleiben Darstellung, Interaktion und Übergänge synchron. Das Modell muss keine fertigen Bilder halluzinieren und die Spiellogik muss Objekte nicht nachträglich aus Pixeln rekonstruieren.

## Repräsentation

Eine Außenwelt besteht aus einem parametrischen Terrain Graph und maximal acht kanonisch geordneten Slots. Jeder Slot trägt Presence, Klasse, Position, Aktion und Trigger.

Der Terrain Graph enthält:

```text
Biom + Küstenrichtung + Uferlinie + Strandbreite + Felsigkeit
```

Die Größen sind weiterhin an die Klasse gebunden:

| Klasse | Breite | Höhe |
|---|---:|---:|
| `tree` | 5 | 8 |
| `rock` | 5 | 4 |
| `npc` | 5 | 9 |
| `portal` | 6 | 8 |

## Modellköpfe

| Kopf | Aufgabe | Ausgabe |
|---|---|---|
| Terrain | Landschaftsstruktur | Parameter und Kategorien |
| Position | absolute Slotposition | X/Y je Slot |
| Presence | Slot vorhanden | binäres Logit je Slot |
| Class | Objektart über Attribute Encoder | vier Klassen |
| Action | mögliche Aktion über Attribute Encoder | drei Klassen |
| Trigger | Übergangstyp über Attribute Encoder | vier Klassen |

Presence und Attribute besitzen eigene Encoder. Dadurch konkurrieren weder Anwesenheit noch Klasse, Aktion und Trigger mit dem stärker gewichteten Positions-Loss.

## Maps

- **Semantic Map:** semantische Klasse jedes Pixels
- **Object Map:** eindeutige Slot-ID jedes Objektpixels
- **Interaction Map:** interaktive Pixelmaske

Beim Anklicken eines Pixels löst die Object Map zunächst die Slot-ID auf. Aktion und Trigger bestimmen die Reaktion. Die Folgewelt-ID entsteht deterministisch aus Welt-Seed, Slot-ID, Trigger-Typ und Story-State.

## Bewertung des Referenzlaufs

Die Geometry- und Presence-Pfade übertragen sich stabil auf acht variable Slots. Klassen, Aktionen und Trigger werden überwiegend richtig erkannt, sind aber noch deutlich von vollständiger Zuverlässigkeit entfernt. Kleine Klassen verlieren bei einer falschen Klasse oder einem Pixel Versatz überproportional IoU.

Der Seed-Token-Kopf aus 0.5 ist entfernt. Ein ordinal vorhergesagter Identitätswert war semantisch fragwürdig: Benachbarte Token sind nicht automatisch ähnliche Folgewelten. Seit 0.5.1 werden Übergänge deterministisch berechnet. 0.5.2 trennt zusätzlich den Attribute Encoder vom Geometry Encoder.

## Aktuelle Grenzen

- kanonische Slot-Reihenfolge statt Matching
- maximal acht Objekte
- feste Objektgrößen und rechteckige Masken
- synthetischer Datengenerator statt kuratierter Editor-Daten
- noch keine Beziehungen, Animationen oder persistenter World State

## Nächste Architekturarbeit

Nach der Validierung verzweigter Story-States folgt permutation-invariantes Matching. Animationen werden später als Zustand des Scene Graphs modelliert; der Renderer wählt wiederverwendbare Sprite-Frames und aktualisiert parallel die Logik-Maps.
