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

- **Terrain Map:** Geländeklasse jedes Pixels
- **Semantic Map:** semantische Klasse jedes Pixels
- **Object Map:** eindeutige Slot-ID jedes Objektpixels
- **Walkability Map:** begehbare und blockierte Terrainpixel
- **Interaction Map:** interaktive Pixelmaske

Beim Anklicken eines Pixels löst die Object Map zunächst die Slot-ID auf. Aktion und Trigger bestimmen die Reaktion. Die Folgewelt-ID entsteht deterministisch aus Welt-Seed, Slot-ID, Trigger-Typ und Story-State.

## Verschachtelte Weltarchitektur

Städte und Dörfer werden nicht als einzelne Object Slots modelliert. Sie bilden eine eigene Ebene zwischen Terrain und Gebäuden:

```text
World Scene Graph
├─ Terrain Layer
│  ├─ Biom, Wasser, Strand und Land
│  └─ Vegetations- und Felsregionen
├─ Settlement Layer
│  ├─ Dorf, Stadt, Hafen oder Ruine
│  └─ Zentrum, Größe, Dichte und Stil
├─ District Layer
│  ├─ Wohnen, Markt, Hafen und Industrie
│  └─ Straßennetz und Grundstücke
├─ Building Layer
│  ├─ Gebäudeart, Eingang und Funktion
│  └─ Außenform und Kollisionsmaske
└─ Interior Layer
   ├─ Raumstruktur und Türen
   └─ NPCs, Gegenstände und Portale
```

Jede Ebene besitzt einen eigenen Scene Graph und Renderer. Dadurch muss das Modell nicht jedes Haus, jeden Baum oder jeden Straßenpixel einzeln vorhersagen.

## Deterministische Hierarchie

Übergänge zwischen den Ebenen werden aus stabilen Identitäten abgeleitet:

```text
Welt-Seed + Settlement-ID + Story-State → Siedlungs-Seed
Siedlungs-Seed + Gebäude-ID             → Innenraum-Seed
Innenraum-Seed + Object-ID              → Folgewelt-Seed
```

Ein Dorf bleibt dadurch bei jedem Besuch dasselbe Dorf. Story-State kann kontrolliert Veränderungen wie zerstörte Gebäude, neue Bewohner oder gesperrte Wege einbringen.

## Aktuelle Grenzen

- kanonische Slot-Reihenfolge statt Matching
- maximal acht Objekte
- feste Objektgrößen und rechteckige Masken
- synthetischer Datengenerator statt kuratierter Editor-Daten
- noch keine Settlement-, District- oder Building-Layer
- noch keine Beziehungen, Animationen oder persistenter World State

## Nächste Architekturarbeit

0.6.1 führt terrainrelative Objektpositionen und deterministische Vegetation ein. Ab 0.7 folgen Dörfer und Städte als Settlement Layer. Die detaillierte Reihenfolge steht in [`roadmap.md`](roadmap.md).
