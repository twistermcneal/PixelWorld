# PixelWorld

PixelWorld erforscht einen generativen Weltbaukasten, der aus `Prompt + Seed` einen strukturierten Scene Graph erzeugt und daraus sichtbare Pixel sowie pixelgenaue Logik-Maps rendert.

## Aktueller Stand

**Version 0.6 – Meilenstein 3: Landschaften und Terrain**

Das Modell erzeugt strukturierte Außenwelten mit einem Terrain Graph und bis zu acht variablen Object Slots. Jeder vorhandene Slot enthält:

- Objektklasse: `tree`, `rock`, `npc` oder `portal`
- absolute X/Y-Position in der Landschaft
- Aktion: `LOOK`, `USE` oder `SCAN`
- Trigger: `NONE`, `WORLD`, `STORY` oder `SECRET`
- deterministisch abgeleitete Folgewelt-ID

Der Terrain Graph beschreibt Biome, Küstenrichtung, Uferlinie, Strandbreite und Felsigkeit. Der deterministische Rasterizer erzeugt daraus Terrain-, Semantic-, Object-, Walkability- und Interaction-Maps.

## Ergebnis des 0.5-Referenzlaufs

| Metrik | Ergebnis |
|---|---:|
| Presence Accuracy | 0,981 |
| relatives Positions-MAE | 0,495 px |
| absolutes Positions-MAE | 0,578 px |
| Klassen-Accuracy | 0,759 |
| Aktions-Accuracy | 0,697 |
| Trigger-Accuracy | 0,771 |
| Interaction IoU | 0,686 |
| Seed-Token MAE | 65,1 |
| Seed-Token Exact Accuracy | 0,0 |

Der 0.5-Referenzlauf bestätigt variable Slots und interaktive Metadaten. 0.5.1 ersetzte den gescheiterten Seed-Token-Kopf durch deterministische Übergänge. In 0.5.2 erhalten Klasse, Aktion und Trigger einen eigenen Attribute Encoder und Slot-Decoder, damit sie nicht mehr mit der Geometrie konkurrieren.

## Schnellstart

Voraussetzung ist Python 3.10 oder neuer. Eine CUDA-fähige PyTorch-Installation beschleunigt das Training, ist aber nicht zwingend erforderlich.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Danach [`notebooks/PixelWorld_0_6.ipynb`](notebooks/PixelWorld_0_6.ipynb) öffnen und die Zellen der Reihe nach ausführen. Die Notebooks der 0.5-Reihe bleiben als Vergleiche erhalten.

Das Referenzexperiment verwendet 10.000 synthetische Welten, Batchgröße 128 und 40 Epochen. Die Laufzeit hängt stark von der verfügbaren Hardware ab.

## Architektur

```text
Prompt + Seed
├─ Terrain Encoder → Biom, Küste, Strand und Felsigkeit
├─ Geometry Encoder → absolute Slotpositionen
├─ Presence Encoder → vorhandene Slots
└─ Attribute Encoder → Klasse, Aktion und Trigger
                       ↓
              strukturierter Scene Graph
                       ↓
              deterministischer Rasterizer
                       ↓
      Semantic Map + Object Map + Interaction Map
```

Weitere Details stehen in [`docs/architecture.md`](docs/architecture.md), die Entwicklungsschritte in [`CHANGELOG.md`](CHANGELOG.md).

## Reproduzierbarkeit

- Standard-Seed: `42`
- Weltgröße: `64 × 64` Pixel
- maximale Slotzahl: `8`
- ordinale Koordinatenklassifikation über 65 Pixelklassen
- Objektpositionen relativ zum vorhergesagten Raum
- aktuell feste Objektgrößen pro Klasse

Das Notebook enthält Generator, Targets, Modell, Training, Auswertung, Visualisierung und einen interaktiven Pixel-zu-Folgewelt-Prototyp in einer Datei.

## Roadmap

- 0.6-Terrainmetriken und Object Slots auswerten
- deterministische Vegetation und Wälder als Scatter-Layer
- permutation-invariantes Slot-Matching
- freie Pixelmasken und Sprite-IDs
- Beziehungen wie `on`, `inside`, `locked_by` und `leads_to`
- Editor als kuratierte Datenquelle
- Titelbildschirme, Animation States und Pixel-Art-Renderer
- Story-Constraints und dauerhafter World State

## Lizenz

Aktuell wurde noch keine Open-Source-Lizenz festgelegt. Bis eine Lizenzdatei ergänzt wird, bleiben alle Rechte vorbehalten.
