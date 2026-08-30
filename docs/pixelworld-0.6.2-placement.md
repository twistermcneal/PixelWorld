# PixelWorld 0.6.2 Placement-Vorprüfung

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
