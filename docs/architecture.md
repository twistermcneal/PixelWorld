# Architektur von PixelWorld 0.5.1

## Ziel

PixelWorld trennt die Erzeugung einer Welt in zwei Ebenen:

1. Ein lernendes Modell erzeugt einen strukturierten Scene Graph.
2. Ein deterministischer Rasterizer übersetzt diesen Graphen in Pixel- und Logik-Maps.

So bleiben Darstellung, Interaktion und Übergänge synchron. Das Modell muss keine fertigen Bilder halluzinieren und die Spiellogik muss Objekte nicht nachträglich aus Pixeln rekonstruieren.

## Repräsentation

Eine Welt besteht aus einem rechteckigen Raum und maximal acht kanonisch geordneten Slots. Jeder Slot trägt Presence, Klasse, relative Position, Aktion und Trigger.

Die absolute Objektposition wird berechnet als:

```text
absolute Position = Raumursprung + relative Slotposition
```

Die Größen sind in 0.5 noch an die Klasse gebunden:

| Klasse | Breite | Höhe |
|---|---:|---:|
| `door` | 7 | 16 |
| `npc` | 5 | 9 |
| `object` | 4 | 5 |
| `portal` | 6 | 8 |

## Modellköpfe

| Kopf | Aufgabe | Ausgabe |
|---|---|---|
| Room | Raumgeometrie | vier ordinale Koordinaten |
| Position | relative Slotposition | X/Y je Slot |
| Presence | Slot vorhanden | binäres Logit je Slot |
| Class | Objektart | vier Klassen |
| Action | mögliche Aktion | drei Klassen |
| Trigger | Übergangstyp | vier Klassen |

Presence besitzt einen eigenen Encoder. Dadurch konkurriert die Anwesenheitserkennung nicht mit dem stärker gewichteten Positions-Loss.

## Maps

- **Semantic Map:** semantische Klasse jedes Pixels
- **Object Map:** eindeutige Slot-ID jedes Objektpixels
- **Interaction Map:** interaktive Pixelmaske

Beim Anklicken eines Pixels löst die Object Map zunächst die Slot-ID auf. Aktion und Trigger bestimmen die Reaktion. Die Folgewelt-ID entsteht deterministisch aus Welt-Seed, Slot-ID, Trigger-Typ und Story-State.

## Bewertung des Referenzlaufs

Die Geometry- und Presence-Pfade übertragen sich stabil auf acht variable Slots. Klassen, Aktionen und Trigger werden überwiegend richtig erkannt, sind aber noch deutlich von vollständiger Zuverlässigkeit entfernt. Kleine Klassen verlieren bei einer falschen Klasse oder einem Pixel Versatz überproportional IoU.

Der Seed-Token-Kopf aus 0.5 ist entfernt. Ein ordinal vorhergesagter Identitätswert war semantisch fragwürdig: Benachbarte Token sind nicht automatisch ähnliche Folgewelten. 0.5.1 berechnet Übergänge stattdessen deterministisch und exakt reproduzierbar.

## Aktuelle Grenzen

- kanonische Slot-Reihenfolge statt Matching
- maximal acht Objekte
- feste Objektgrößen und rechteckige Masken
- synthetischer Datengenerator statt kuratierter Editor-Daten
- noch keine Beziehungen, Animationen oder persistenter World State

## Nächste Architekturarbeit

Nach der Validierung verzweigter Story-States folgt permutation-invariantes Matching. Animationen werden später als Zustand des Scene Graphs modelliert; der Renderer wählt wiederverwendbare Sprite-Frames und aktualisiert parallel die Logik-Maps.
