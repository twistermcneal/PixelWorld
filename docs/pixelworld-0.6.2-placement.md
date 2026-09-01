# PixelWorld 0.6.2 Placement-Vorprüfung

## Isolierte Gradientenkonflikt-Ablation

Die bestehende Variante D bleibt unverändert. Zwei D-basierte Diagnosemodi können
über `--gradient-mode` gewählt werden:

- `measure` (`D-MEASURE`) misst pro Batch den Konflikt zwischen
  `2 * (Lregion + Lanchor)` und `Loffset`, verändert aber keine Gradienten.
- `pcgrad` (`D-PCGRAD`) projiziert ausschließlich den Offsetgradienten im gemeinsam
  verwendeten Placement-Encoder und -Decoder, wenn sein Skalarprodukt mit dem
  diskreten Gradienten negativ ist. Offset-, Region- und Anchor-Heads sowie alle
  fremden Modellpfade behalten ihre normalen Gradienten.

Beispiele für kleine, getrennt benannte Vorprüfungen:

```powershell
python -m pixelworld.cli train --version 0.6.2 --variant D --gradient-mode measure --samples 32 --batch-size 8 --epochs 1 --run-id v062-D-MEASURE-smoke --device cpu
python -m pixelworld.cli train --version 0.6.2 --variant D --gradient-mode pcgrad --samples 32 --batch-size 8 --epochs 1 --run-id v062-D-PCGRAD-smoke --device cuda
```

Trainingshistorie und Recovery-Checkpoints enthalten den Modus sowie
epochengemittelte Kosinus-, Norm-, Konflikt- und Projektionsstatistiken. Checkpoints
unterschiedlicher Modi sind absichtlich inkompatibel. Generator, Targets,
Rasterizer, Modellgröße und Loss-Gewichte bleiben identisch zu D.

### Gerichtete Query-Detach-Ablation

`qdet-measure` (`D-QDET-MEASURE`) und `qdet-pcgrad` (`D-QDET-PCGRAD`)
verwenden dieselben Query-Werte und dieselbe einzelne Query-Matrix wie D. Nur die
Placement-Ansicht wird mit `slot_queries.weight.detach()` an den Placement-Decoder
gegeben. Der Attribute-Decoder verwendet weiterhin `slot_queries.weight` mit
normalem Gradientenfluss. Damit können Attribute-Losses die Queries trainieren,
Placement-Losses jedoch nicht.

`qdet-measure` misst Konflikte ohne Projektion. `qdet-pcgrad` projiziert bei einem
negativen Konflikt weiterhin ausschließlich den Offsetgradienten in Placement-Encoder
und Placement-Decoder. Parameterzahl, Heads, Targets, Rasterizer und Loss-Gewichte
ändern sich nicht.

```powershell
python -m pixelworld.cli train --version 0.6.2 --variant D --gradient-mode qdet-measure --samples 32 --batch-size 8 --epochs 1 --run-id v062-D-QDET-MEASURE-smoke --device cpu
python -m pixelworld.cli train --version 0.6.2 --variant D --gradient-mode qdet-pcgrad --samples 32 --batch-size 8 --epochs 1 --run-id v062-D-QDET-PCGRAD-smoke --device cuda
```

PixelWorld 0.6.1 und Variante A bleiben unverändert bei sechs Latentwerten je Slot.
Die Varianten B–E verwenden acht explizite Werte in dieser Reihenfolge:

`presence, class, region, anchor, offset_x, offset_y, action, trigger`

Damit hat das 0.6.2-Layout 71 Werte: sieben Terrain-Latents plus acht Slots mit
je acht Werten. Die beiden Offsetwerte werden direkt von `[0, 1]` nach `[-1, 1]`
abgebildet. Sie beschreiben mit einem festen Radius von acht Pixeln eine lokale
Wunschverschiebung relativ zur kanonischen Anchor-Position.

## Regionsrelative Anchors

Die 16 Anchors bilden ein festes 4×4-Raster innerhalb der Bounding Box aller
gültigen Kandidaten der tatsächlich verwendeten Region. Fehlen Kandidaten in der
angeforderten Region, wird zuerst die bestehende Region-Fallback-Reihenfolge
aufgelöst. Der kanonische Anchor ist der gültige Kandidat mit dem kleinsten
quadratischen Abstand zur normalisierten Rasterposition; Gleichstände werden
nach `y`, dann `x` aufgelöst.

Der Rasterizer projiziert `anchor + 8 * offset` auf die nächste gültige und
kollisionsfreie Position. Terrain-, Wasser-, Weltgrenzen- und Kollisionsregeln
bleiben dabei zwingend. Region-Fallbacks und kollisionsbedingte Sprünge sind
Rasterizer-Effekte und werden nicht Bestandteil des Offset-Lernziels.

## Varianten

- B: stabile Anchors, Vorhersage verwendet Offset null und besitzt keinen Offset-Head.
- C: Offset-Head auf vom Placement-Pfad abgetrennten Slot-Repräsentationen.
- D: Offset-Head mit gemeinsamem Gradient in den Placement-Pfad.
- E: wie C, zusätzlich ein direkt überwachter normalisierter X/Y-Auxiliary-Head
  mit Gradient in den Placement-Pfad.

## Clipping und Projektion

Die frühere weltrelative Snap-Offset-Definition verfehlte das vorgegebene
Clipping-Kriterium bereits strukturell. Die neuen expliziten Latenttargets werden
nicht nachträglich geclippt: Sie liegen per Definition immer in `[-1, 1]`.
Projektionsdistanz, realisierte gegenüber gewünschter Verschiebung und der Anteil
exakt realisierter Wünsche werden getrennt als Rasterizer-Metriken ausgewiesen.
So werden Lernfehler und geometrisch notwendige Projektion nicht vermischt.

Vollständige Mehrseed-Trainingsläufe dürfen erst nach bestandener struktureller
Vorprüfung gestartet werden.

## Reproduzierbarkeit und Artefaktsicherheit

Der Study-Runner startet nur aus einem sauberen Git-Arbeitsbaum. Commit und
Dirty-Status werden vor jedem Trainingsprozess und vor jeder Wiederverwendung
erneut kontrolliert. Run-Summaries, Recovery- und finale Checkpoints sowie die
Studienkonfiguration speichern Commit, Branch, Python-, PyTorch- und
CUDA-Informationen, Device/GPU, Modellparameterzahl, vollständige Konfiguration,
Evaluationsseeds und den gemeinsamen B–E-Target-Digest.

Der kanonische Digest der 14.000 B–E-Zielwelten lautet:

`a04645d1b56d45b4916e496bee83cbb2837da726184fa7a4a2f269046434c5ae`

Analyse-Caches werden nur bei exakter Übereinstimmung von Schema,
Generator-/Target-Version, Dimensionskonstanten, Radius, Samplezahl und Digest
wiederverwendet. Run-Verzeichnisse und jede einzelne Artefaktdatei werden vor
dem Zugriff erneut kanonisch aufgelöst; Symlink-, Junction-, Gerätenamen- und
Traversal-Escapes werden abgewiesen.

## Bedeutung des Baseline-Vergleichs

Variante A verwendet die eingefrorene 0.6.1-Generatorsemantik. B–E teilen die
neuen 8-Latent-Zielwelten. Die Datei
`seed_matched_benchmark_deltas.csv` vergleicht A und B–E daher nur nach gleichem
Trainingsseed. Sie enthält keine gepaarten Target-Welt-Differenzen. Statistisch
gemeinsam erzeugte Target-Welten dürfen ausschließlich für B–E behauptet werden.

## Isolierte Split-Query-Ablation

`D-SPLIT-MEASURE` (`--gradient-mode split-measure`) besitzt eine eigene trainierbare
`placement_slot_queries`-Matrix, während `slot_queries.weight` exklusiv dem
Attributpfad gehört. Die Placement-Matrix wird nach der vollständigen bisherigen
Modellinitialisierung ohne weiteren Zufallszug als bitgenauer Klon angelegt. Damit
bleiben alle bisherigen Parameter und Forward-Werte vor dem Training identisch zu D.
Die Parameterzahl steigt ausschließlich um `8 * 320 = 2.560`.

Der Modus misst, projiziert aber keine Gradienten. Er protokolliert
Region-vs.-Anchor, Offset-vs.-Region, Offset-vs.-Anchor und
Offset-vs.-kombiniertem Region+Anchor getrennt für Placement-Encoder,
Placement-Decoder, Placement-Queries und den Gesamtpfad. Dazu kommen negative
Konfliktraten und aufgabenspezifische Gradientennormen. Checkpoints tragen
`split_placement_queries=true` und eine versionierte `query_schema_version`; sie
sind nicht mit D-, PCGrad- oder QDET-Checkpoints austauschbar. Vollständige
Trainingsläufe bedürfen einer gesonderten Freigabe.
