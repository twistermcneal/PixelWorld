# PixelWorld 0.6.3: LLM-Adventure-Architektur

## Ziel und Implementierungsplan

0.6.3 beginnt einen neuen, von den reproduzierbaren 0.6.1- und 0.6.2-Forschungspfaden getrennten vertikalen Prototyp. Die langfristige Pipeline lautet:

```text
Benutzerprompt → Story Director → AdventureSpec → Game Compiler → RoomSpec
→ Scene Graph → visuelle Szene → deterministische Runtime → Adventure
```

Phase 1 wurde in diesen kleinen Schritten aufgebaut: strikte Datenverträge und Ontologie, deterministischer Compiler und Vektornavigation, Runtime und Solver, Vorabvalidierung, Browserexport, CLI sowie End-to-End-Tests. Die neue Implementierung liegt vollständig unter `pixelworld/adventure/`; die historischen Generator-, Trainings- und Auswertungspfade werden nicht importiert oder verändert.

## Scope von Phase 1

Phase 1 besitzt genau einen Raum, einen Spieler, einen NPC, ein Inventar, ein mehrschrittiges Kombinationsrätsel und einen Erfolgszustand. Enthalten sind Walkboxes, Kollisionen, Hotspots, Walk-to-Punkte, Vordergrundokklusion und ein offline nutzbarer Browserexport.

Nicht enthalten sind mehrere Räume, Netzwerkzugriffe, Audio, freie Codegenerierung, echtes LLM-Inferencing, Bildgenerierungs-APIs, neuronales Training und komplexe Dialogbäume. Das Director-Fixture ist ausdrücklich Testdatenquelle und kein simuliertes oder angebliches LLM-Ergebnis.

## Verträge und Sicherheitsgrenzen

Alle JSON-Verträge tragen `schema_version: "0.6.3"`. `AdventureSpec` umfasst Titel, Prämisse, Ton, visuelles Theme, Spieler, Figuren, Orte, Objekte, Inventarobjekte, Ziele, Rätsel, Interaktionen und Endbedingungen. Die Validierung lehnt unbekannte Felder rekursiv ab. IDs entsprechen `^[a-z][a-z0-9_]{0,63}$`, sind je Namensraum eindeutig und werden referenziell geprüft. Damit können IDs weder Pfade noch Shellfragmente sein.

Interaktionsbedingungen bestehen nur aus `equals`, `inventory_contains` und `inventory_missing`; Effekte nur aus `set`, `inventory_add` und `inventory_remove`. Zustandszugriffe sind auf `objects`, `objectives` und `flags` begrenzt. Weder Python-Ausdrücke noch JavaScript, Shellcode, Templates oder dynamische Importe werden aus der Spec ausgeführt.

`StoryDirector` ist eine provider-neutrale abstrakte Schnittstelle. `FixtureStoryDirector` liefert die kuratierte Golden-Spec, `JsonStoryDirector` lädt eine vorhandene Spec. Ein späterer OpenAI- oder lokaler Adapter muss lediglich dieselbe Schnittstelle implementieren. Seine Ausgabe durchläuft unverändert die strikte Validierung; Compiler und Runtime benötigen keine Providerkenntnis.

## Theme-Ontologie

Die versionierte Ontologie enthält `mad_scientist_lab`, `pirate_harbor`, `forest_ruin`, `spaceship` und `medieval_village`. Jedes Theme definiert erlaubte Terrain-, Architektur- und Objektklassen, NPC-Archetypen, Palette, Lichtstimmungen, Portal- und Vordergrundtypen sowie Hotspotrollen.

Ein unbekanntes Theme, eine abweichende Ortstheme oder eine inkompatible Objekt-/NPC-Klasse ist ein verständlicher Fehler. Der Compiler erfindet keine Ersatzklasse und verwendet keinen stillen Fallback.

## Compiler, RoomSpec und Scene Graph

Der Compiler nimmt ausschließlich eine validierte AdventureSpec entgegen. Er erzeugt ein RoomSpec mit Pflichtentitäten, Ausgang, Spielerstart und Zielen sowie einen Scene Graph mit Hintergrundebenen, semantischen Regionen, Walkboxes, Navigationskanten, Kollisions- und Okklusionspolygonen, Entitäten, Hotspots, Walk-to-Punkten, Z-Ebenen, Portalen und Initialzustand.

Phase 1 verwendet eine feste logische Größe von **128 × 72**. 64 × 64 bleibt sinnvoll für die bisherigen Außenweltstudien, bietet in einem Innenraum mit 28 Pixel breiter zentraler Maschine aber zu wenig seitlichen Navigationsraum und zu kleine, voneinander unterscheidbare Hotspots. 128 × 72 hält ein 16:9-Browserbild, erlaubt Vordergrundstaffelung und bleibt klein genug für echte Pixel-Art.

Die kanonische JSON-Serialisierung des kompilierten Spiels erhält einen SHA-256-`compile_digest`. Identische Eingaben erzeugen identische Daten und denselben Digest.

## Vektornavigation

Walkboxes sind konvexe Polygone. Der Navigator implementiert Point-in-Polygon inklusive Rand, deterministische Projektion auf das nächstgelegene Polygon, einen expliziten Nachbarschaftsgraphen und eine kürzeste Route mit lexikografischen Tie-Breaks. Eine Sichtlinienglättung entfernt unnötige Zwischenpunkte nur dann, wenn das gesamte Segment innerhalb der Walkboxvereinigung und außerhalb aller Kollisionen bleibt.

Der Validator prüft jeden Walk-to-Punkt und berechnet vom Start eine Route zu jeder Pflichtentität und zum Portal. Die Zeitmaschine bildet eine eigene Kollisionsfläche; Wege können sie nur über die vordere Walkbox umrunden.

## Runtime und Zustandsmodell

`AdventureRuntime` verwaltet Spielerposition, sortiertes Inventar, Objektzustände, Zielzustände, Flags und Abschlussstatus. Unterstützt werden `move_to`, `look_at`, `talk_to`, `take`, `use` und `combine`. Jede Aktion liefert Erfolg, Text, konkrete Zustandsänderungen, Animationshinweis, Bewegungsroute und die danach verfügbaren Aktionen.

Save/Load serialisiert ausschließlich das validierte Runtime-Schema. Version, Feldmenge, Spielerposition, Inventar-IDs sowie Objekt- und Zielnamensräume werden beim Laden geprüft.

## Validator und Solver

Vor Spielstart prüft der Validator Spec, Versionen, Compile-Digest, RoomSpec, Scene Graph, Polygongeometrie, Weltgrenzen, Kollisionsüberschneidungen, Pflichtplatzierungen, Hotspots, Walk-to-Punkte, Pfaderreichbarkeit, Portal und Erfolgsbedingung. Anschließend führt er den Solver aus.

Der Solver ist eine begrenzte Breitensuche über zulässige deklarative Interaktionen. Zustände werden über kanonisches JSON dedupliziert, Aktionen durch stabile Interaktions-IDs sortiert und die kürzeste Lösung samt Digest dokumentiert. Das Zustandslimit verhindert unbeschränkte Suche; eine nicht lösbare Spec macht die Gesamtvalidierung ungültig.

## Browserexport

Der Export besteht aus statischem HTML, CSS und JavaScript ohne Framework, CDN oder Netzwerkzugriff. Die Canvas-Szene rendert eigene geometrische Pixel-Platzhalter, Entitäten nach Klasse, Spieler, Inventar, Hotspotauswahl, kontextuelle Aktionen, Portal-/Maschinenzustand sowie umschaltbare Walkbox-, Kollisions- und Walk-to-Debugdaten.

Die JavaScript-Runtime interpretiert `runtime_rules` und den Scene Graph generisch. Golden-Room-IDs steuern keine Sonderzweige. Das kompilierte Spiel wird als Datenkonstante eingebettet, weshalb `index.html` auch per `file://` geöffnet werden kann. Ein lokaler HTTP-Server ist für reproduzierbare Browser-Smoke-Tests dennoch vorzuziehen.

## Spätere Entwicklung

Ein echtes LLM soll zunächst nur AdventureSpec-Kandidaten erzeugen. Validierungsfeedback kann für Reparaturversuche an den Provider zurückgegeben werden, niemals jedoch als ausführbarer Code. Danach können mehrere Räume, umfangreichere typisierte Prädikate, Dialogdaten, Autorenwerkzeuge und hochwertige eigene Assets folgen.

Aus validierten Specs, Compilerentscheidungen, Solverpfaden und Rasterisierungen lassen sich später synthetische Trainingspaare bilden. Ein trainiertes PixelWorld-Modell kann dann Layoutvorschläge, Walkboxen, Asset-Varianten oder visuelle Komposition vorhersagen. Referenzprüfung, Zustandsmaschine, Kollision, Solver, Sicherheitsgrenzen und Export bleiben deterministisch und außerhalb des Modells.

