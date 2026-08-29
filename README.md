# PixelWorld

PixelWorld erforscht einen generativen Weltbaukasten, der aus `Prompt + Seed` einen strukturierten Scene Graph erzeugt und daraus sichtbare Pixel sowie pixelgenaue Logik-Maps rendert.

## Aktueller Stand

**Version 0.5 – Meilenstein 2 (Experiment)**

Das Modell erzeugt einen Raum mit bis zu acht variablen Object Slots. Jeder vorhandene Slot enthält:

- Objektklasse: `door`, `npc`, `object` oder `portal`
- relative X/Y-Position im Raum
- Aktion: `LOOK`, `USE` oder `SCAN`
- Trigger: `NONE`, `WORLD`, `STORY` oder `SECRET`
- Seed-Token für deterministische Folgewelten

Der Scene Graph wird deterministisch in Semantic-, Object- und Interaction-Maps gerastert. Ein Klick kann dadurch einem Objekt und dessen Folgewelt zugeordnet werden, ohne Logik aus dem Bild zurückraten zu müssen.

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

Meilenstein 2 bestätigt variable Slots und interaktive Metadaten. Der Seed-Token-Kopf ist noch nicht brauchbar und wird ausdrücklich nicht als gelöst betrachtet.

## Schnellstart

Voraussetzung ist Python 3.10 oder neuer. Eine CUDA-fähige PyTorch-Installation beschleunigt das Training, ist aber nicht zwingend erforderlich.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Danach [`notebooks/PixelWorld_0_5.ipynb`](notebooks/PixelWorld_0_5.ipynb) öffnen und die Zellen der Reihe nach ausführen.

Das Referenzexperiment verwendet 10.000 synthetische Welten, Batchgröße 128 und 40 Epochen. Die Laufzeit hängt stark von der verfügbaren Hardware ab.

## Architektur

```text
Prompt + Seed
├─ Geometry Encoder → Raum und relative Slotpositionen
├─ Presence Encoder → vorhandene Slots
└─ Slot Decoder → Klasse, Aktion, Trigger und Seed-Token
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

- Seed-Token aus der Modellvorhersage herauslösen oder über einen eigenen Pfad lernen
- permutation-invariantes Slot-Matching
- freie Pixelmasken und Sprite-IDs
- Beziehungen wie `on`, `inside`, `locked_by` und `leads_to`
- Editor als kuratierte Datenquelle
- Titelbildschirme, Animation States und Pixel-Art-Renderer
- Story-Constraints und dauerhafter World State

## Lizenz

Aktuell wurde noch keine Open-Source-Lizenz festgelegt. Bis eine Lizenzdatei ergänzt wird, bleiben alle Rechte vorbehalten.
