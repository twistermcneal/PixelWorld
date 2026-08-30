# PixelWorld

PixelWorld erforscht einen generativen Weltbaukasten, der aus `Prompt + Seed` einen strukturierten Scene Graph erzeugt und daraus sichtbare Pixel sowie pixelgenaue Logik-Maps rendert.

## Aktueller Stand

**Version 0.6.1 – Terrainregionen und Vegetation**

Das Modell erzeugt strukturierte Außenwelten mit Terrain- und Region Graph sowie bis zu acht wichtigen Landmark Slots. Jeder vorhandene Slot enthält:

- Objektklasse: `chest`, `npc`, `portal` oder `ruin`
- Terrainregion: Strand, offenes Land, Felsfeld oder Wald
- einen von 16 kanonischen Anchors innerhalb der Region
- Aktion: `LOOK`, `USE` oder `SCAN`
- Trigger: `NONE`, `WORLD`, `STORY` oder `SECRET`
- deterministisch abgeleitete Folgewelt-ID

Der Terrain Graph beschreibt Biome, Küstenrichtung, Uferlinie, Strandbreite, Felsigkeit, Waldstufe und Vegetationsdichte. Normale Bäume werden deterministisch verteilt und verbrauchen keine Landmark Slots.

## Ergebnis des 0.6-Referenzlaufs

| Metrik | Ergebnis |
|---|---:|
| Terrain Mean IoU | 0,970 |
| Biome Accuracy | 1,000 |
| Orientation Accuracy | 1,000 |
| Terrain Parameter-MAE | 0,144 px |
| Presence Accuracy | 0,983 |
| Klassen-Accuracy | 0,937 |
| Aktions-Accuracy | 0,928 |
| Trigger-Accuracy | 0,935 |
| absolutes Positions-MAE | 1,953 px |
| Interaction IoU | 0,470 |

Der 0.6-Referenzlauf bestätigt den Terrain Graph. 0.6.1 ersetzt die schwache absolute Positionsvorhersage durch `Terrainregion + Anchor` und führt deterministische Vegetation ein.

Der vollständige Benchmark mit Trainingskurve und Visualisierung liegt unter [`results/0.6`](results/0.6/README.md).

## Schnellstart

Voraussetzung ist Python 3.10 oder neuer. Eine CUDA-fähige PyTorch-Installation beschleunigt das Training, ist aber nicht zwingend erforderlich.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Danach [`notebooks/PixelWorld_0_6_1.ipynb`](notebooks/PixelWorld_0_6_1.ipynb) öffnen und die Zellen der Reihe nach ausführen. Das [`0.6-Notebook`](notebooks/PixelWorld_0_6.ipynb) bleibt als Vergleich erhalten.

Das 0.6.1-Referenzexperiment verwendet 14.000 synthetische Landschaften, Batchgröße 128 und 45 Epochen. Die Laufzeit hängt stark von der verfügbaren Hardware ab.

Beim Start werden die Landschaften einmalig auf der CPU vorberechnet. Anschließend trainiert das Modell automatisch auf CUDA, sofern `torch.cuda.is_available()` wahr ist. Das Notebook gibt das erkannte Gerät und den GPU-Namen aus.

## Architektur

```text
Prompt + Seed
├─ Terrain Encoder → Biom, Küste, Strand und Felsigkeit
├─ Placement Encoder → Terrainregion und Anchor
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
- terrainrelative Landmark-Positionen über Region und Anchor
- aktuell feste Objektgrößen pro Klasse

Das Notebook enthält Generator, Targets, Modell, Training, Auswertung, Visualisierung und einen interaktiven Pixel-zu-Folgewelt-Prototyp in einer Datei.

## Roadmap

- **0.6:** Landschaft und Terrain
- **0.6.1:** terrainrelative Positionen, Vegetation und Wälder – in Auswertung
- **0.7:** Settlement Layer für Dörfer
- **0.7.1:** Straßen, Grundstücke und Gebäude
- **0.7.2:** Stadtbezirke und größere Städte
- **0.8:** Übergänge Landschaft → Stadt → Gebäude → Innenraum

Die vollständige Planung steht in [`docs/roadmap.md`](docs/roadmap.md).

## Lizenz

Aktuell wurde noch keine Open-Source-Lizenz festgelegt. Bis eine Lizenzdatei ergänzt wird, bleiben alle Rechte vorbehalten.
