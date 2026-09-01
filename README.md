# PixelWorld

PixelWorld erforscht einen generativen Weltbaukasten, der aus `Prompt + Seed` einen strukturierten Scene Graph erzeugt und daraus sichtbare Pixel sowie pixelgenaue Logik-Maps rendert.

## Aktueller Stand

**Version 0.6.3 – erster strukturierter Adventure-Room-Prototyp**

0.6.3 ergänzt den von den reproduzierbaren 0.6.1-/0.6.2-Forschungspfaden getrennten Golden Room „Professor Knallberts chronochemisches Labor“. Eine streng validierte AdventureSpec wird deterministisch in RoomSpec, Scene Graph, Navigation, Runtime und einen offline spielbaren Browserexport kompiliert. Einstieg und Sicherheitsgrenzen beschreibt [`docs/pixelworld-0.6.3-llm-adventure-design.md`](docs/pixelworld-0.6.3-llm-adventure-design.md), der Raum selbst steht in [`docs/pixelworld-0.6.3-golden-room.md`](docs/pixelworld-0.6.3-golden-room.md).

```powershell
python -m pixelworld.cli adventure-generate --version 0.6.3 --director fixture --prompt "Ein verrückter Wissenschaftler repariert seine Zeitmaschine" --output outputs/adventures/0.6.3-golden-lab
python -m pixelworld.cli adventure-generate --version 0.6.3 --director fixture --fixture pirate_harbor --output outputs/adventures/0.6.3-pirate-harbor
python -m pixelworld.cli adventure-validate --spec outputs/adventures/0.6.3-golden-lab/adventure_spec.json
python -m pixelworld.cli adventure-solve --game outputs/adventures/0.6.3-golden-lab/game.json
```

Phase 2 ergänzt einen echten, aber standardmäßig nicht konfigurierten OpenAI-kompatiblen Story Director. Er verwendet ausschließlich strukturierte AdventureSpec-Ausgaben; Compiler, Validator und Solver bleiben die Autorität. Konfiguration, Requestvertrag, Repair-Loop und Secret-Schutz stehen in [`docs/pixelworld-0.6.3-openai-compatible-director.md`](docs/pixelworld-0.6.3-openai-compatible-director.md). Die Teststrategie führt keine externen Modellaufrufe aus.

**Reproduzierbare Forschungsbasis: Version 0.6.1 – Terrainregionen und Vegetation**

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

## Core-Paket und CLI

PixelWorld 0.6.1 ist als wiederverwendbares Python-Paket organisiert. Notebook und CLI verwenden denselben Core für Weltgenerierung, Placement, Modell, Training, Evaluation und Inferenz.

Unter Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pixelworld.cli train --version 0.6.1
```

PyTorch muss für CUDA mit einem zur lokalen Hardware passenden CUDA-Wheel installiert sein. Die allgemeine `requirements.txt` erzwingt kein bestimmtes CUDA-Wheel. Wenn `torch.cuda.is_available()` falsch ist, fällt PixelWorld automatisch auf CPU zurück; mit `--device cuda` wird stattdessen ein verständlicher Fehler ausgegeben.

Die wichtigsten Befehle:

```powershell
python -m pixelworld.cli train --version 0.6.1 --samples 2000 --batch-size 128 --epochs 3 --seed 42
python -m pixelworld.cli evaluate --run <run-id>
python -m pixelworld.cli infer --run <run-id> --prompt "tropical coast beach forest rock portal" --seed 500000
python -m pixelworld.cli runs
python -m pixelworld.cli resume --run <run-id>
```

Läufe liegen unter `outputs/runs/<run-id>/`. `latest.pt` enthält den Recovery-Zustand, `final.pt` den abgeschlossenen Lauf. Status, Konfiguration, vollständige Loss-Historie, Evaluation, Laufzeiten und Hardwareinformationen werden daneben als JSON/CSV/Log gespeichert. Checkpoints und Statusdateien werden atomar ersetzt.

Danach kann [`notebooks/PixelWorld_0_6_1.ipynb`](notebooks/PixelWorld_0_6_1.ipynb) als Forschungs- und Visualisierungsoberfläche geöffnet werden. Es importiert den gemeinsamen Core und enthält keine zweite Trainingsimplementierung. Das [`0.6-Notebook`](notebooks/PixelWorld_0_6.ipynb) bleibt als historischer Vergleich erhalten.

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

Der Golden-Test vergleicht alle ungerundeten Loss-Komponenten, zwölf Evaluationsmetriken und jeden Tensor des finalen Modellzustands mit `outputs/0.6.1-reference`. Einrichtung, Recovery und Validierungsbefehle sind in [`docs/core-cli.md`](docs/core-cli.md) dokumentiert.

WebUI, Docker und PixelWorld 0.6.2 sind ausdrücklich spätere, getrennte Schritte.

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
