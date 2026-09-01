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

Alle JSON-Verträge tragen `schema_version: "0.6.3"`. `AdventureSpec` umfasst Titel, Prämisse, Ton, visuelles Theme, Spieler, Figuren, Orte, Objekte, Inventarobjekte, Ziele, Rätsel, Interaktionen, deklarierte Flags und Endbedingungen. Figuren und Objekte tragen außerdem Rolle, bevorzugte Layoutzone und vollständigen Initialzustand; NPC-Dialog und Portalziel sind explizite Daten. Die Validierung lehnt unbekannte Felder rekursiv ab. IDs entsprechen `^[a-z][a-z0-9_]{0,63}$`, sind je Namensraum eindeutig und werden referenziell geprüft. Object- und Character-IDs dürfen sich nicht überschneiden. Eine Inventory-ID darf genau dann einer Object-ID entsprechen, wenn dieses Objekt tragbar ist; jedes tragbare Objekt muss eine entsprechende Inventory-Definition besitzen.

Phase 1 begrenzt nicht vertrauenswürdige Ausgaben fest auf einen Ort, zwei Figuren, 16 Objekte, 16 Inventarobjekte, 16 Interaktionen, acht Ziele, acht Rätsel, vier Endbedingungen und 16 deklarierte Flags. Polygone, Referenz-, Bedingungs- und Effektlisten sind ebenfalls begrenzt. Titel, Namen, Beschreibungen, Dialoge und Interaktionstexte besitzen dokumentierte Längenlimits von 64 bis 800 Zeichen. Zustandsstrings sind maximal 160 Zeichen lang; Zustandszahlen müssen endlich sein und innerhalb ±1.000.000.000 liegen. Koordinaten sind echte, endliche Zahlen – `bool` gilt nicht als Zahl – und werden zusätzlich gegen die 128×72-Raumgrenzen geprüft. Nur die bekannten, flachen Strukturen sind erlaubt; dadurch ist die JSON-Verschachtelung implizit begrenzt.

Interaktionsbedingungen bestehen nur aus `equals`, `inventory_contains` und `inventory_missing`; Effekte nur aus `set`, `inventory_add` und `inventory_remove`. `take`, `use` und `combine` verlangen exakt null, ein beziehungsweise zwei unterschiedliche Item-IDs; ein Combine-Ziel muss ein tragbarer Inventarcontainer sein. `talk_to` und `look_at` haben keine Items, `move_to` ist keine deklarative Interaction. Zustandszugriffe sind auf vorab bekannte Felder in `objects`, `objectives` und deklarierten `flags` begrenzt. Pfad, Operation und Werttyp werden bereits vor dem Kompilieren gegeneinander geprüft. Weder Python-Ausdrücke noch JavaScript, Shellcode, Templates oder dynamische Importe werden aus der Spec ausgeführt.

`StoryDirector` ist eine provider-neutrale abstrakte Schnittstelle. `FixtureStoryDirector` liefert die kuratierte Golden-Spec, `JsonStoryDirector` lädt eine vorhandene Spec. Ein späterer OpenAI- oder lokaler Adapter muss lediglich dieselbe Schnittstelle implementieren. Seine Ausgabe durchläuft unverändert die strikte Validierung; Compiler und Runtime benötigen keine Providerkenntnis.

## Theme-Ontologie

Die versionierte Ontologie enthält `mad_scientist_lab`, `pirate_harbor`, `forest_ruin`, `spaceship` und `medieval_village`. Jedes Theme definiert erlaubte Terrain-, Architektur- und Objektklassen, NPC-Archetypen, Palette, Lichtstimmungen, Portal- und Vordergrundtypen sowie Hotspotrollen.

Ein unbekanntes Theme, eine abweichende Ortstheme oder eine inkompatible Objekt-/NPC-Klasse ist ein verständlicher Fehler. Der Compiler erfindet keine Ersatzklasse und verwendet keinen stillen Fallback.

## Compiler, RoomSpec und Scene Graph

Der Compiler nimmt ausschließlich eine validierte AdventureSpec entgegen. Er erzeugt ein RoomSpec mit Pflichtentitäten, Ausgang, Spielerstart und Zielen sowie einen Scene Graph mit Hintergrundebenen, semantischen Regionen, Walkboxes, Navigationskanten, Kollisions- und Okklusionspolygonen, Entitäten, Hotspots, Walk-to-Punkten, Z-Ebenen, Portalen und Initialzustand. Phase 1 besitzt deterministische, themenspezifische Templates für Labor und Piratenhafen. Innerhalb eines Templates erfolgt die Platzierung nach Theme, Klasse, Hotspotrolle und deklarierter `preferred_zone`; konkrete Entity-IDs beeinflussen die Platzierung nicht. Mehrfach belegbare Zonen verwenden stabile Klassen-/Rollen-/ID-Tie-Breaks.

Phase 1 verwendet eine feste logische Größe von **128 × 72**. 64 × 64 bleibt sinnvoll für die bisherigen Außenweltstudien, bietet in einem Innenraum mit 28 Pixel breiter zentraler Maschine aber zu wenig seitlichen Navigationsraum und zu kleine, voneinander unterscheidbare Hotspots. 128 × 72 hält ein 16:9-Browserbild, erlaubt Vordergrundstaffelung und bleibt klein genug für echte Pixel-Art.

Die kanonische JSON-Serialisierung des kompilierten Spiels erhält einen SHA-256-`compile_digest`. Identische Eingaben erzeugen identische Daten und denselben Digest.

## Vektornavigation

Walkboxes sind konvexe Polygone. Der Navigator implementiert Point-in-Polygon inklusive Rand, deterministische Projektion auf das nächstgelegene Polygon, einen expliziten Nachbarschaftsgraphen und eine kürzeste Route mit lexikografischen Tie-Breaks. Eine Sichtlinienglättung entfernt unnötige Zwischenpunkte nur dann, wenn das gesamte Segment innerhalb der Walkboxvereinigung und außerhalb aller Kollisionen bleibt.

Der Validator prüft jeden Walk-to-Punkt und berechnet vom Start eine Route zu jeder Pflichtentität und zum Portal. Die Zeitmaschine bildet eine eigene Kollisionsfläche; Wege können sie nur über die vordere Walkbox umrunden.

## Runtime und Zustandsmodell

`AdventureRuntime` verwaltet Spielerposition, sortiertes Inventar, Objektzustände, Zielzustände, Flags und Abschlussstatus. Der Compiler liefert dazu ein vollständiges `state_schema` mit allen IDs, Feldern und exakten JSON-Typen. Unterstützt werden `move_to`, `look_at`, `talk_to`, `take`, `use` und `combine`. Look-, Talk- und Interaction-Texte stammen aus kompilierten Daten. Jede Aktion liefert Erfolg, Text, konkrete Zustandsänderungen, Animationshinweis, Bewegungsroute und die danach verfügbaren Aktionen.

Save/Load serialisiert ausschließlich das validierte Runtime-Schema und den Compile-Digest. Version, Digest, sämtliche verschachtelten IDs/Felder/Typen, Flags, Inventar, endliche begehbare Spielerposition und Endzustand werden beim Laden geprüft. `completed` wird gegen die Endbedingungen neu berechnet; ein Save eines anderen Spiels oder ein gefälschter Abschluss wird abgewiesen. Interaktionen wenden alle Effekte auf einer Kopie an, validieren den gesamten Folgezustand und übernehmen ihn erst danach. Ein später fehlschlagender Effekt hinterlässt daher weder in Python noch JavaScript Teiländerungen.

## Validator und Solver

Vor Spielstart prüft der Validator Spec, Versionen, Compile-Digest, RoomSpec, Scene Graph, Polygongeometrie, Weltgrenzen, Kollisionsüberschneidungen, Pflichtplatzierungen, Hotspots, Walk-to-Punkte, Pfaderreichbarkeit, Portal und Erfolgsbedingung. Anschließend führt er den Solver aus.

Der Solver ist eine begrenzte Breitensuche über zulässige deklarative Interaktionen. Zustände werden über kanonisches JSON dedupliziert, Aktionen durch stabile Interaktions-IDs sortiert und die kürzeste Lösung samt Digest dokumentiert. Das Zustandslimit verhindert unbeschränkte Suche; eine nicht lösbare Spec macht die Gesamtvalidierung ungültig.

## Browserexport

Der Export besteht aus statischem HTML, CSS und JavaScript ohne Framework, CDN oder externe Netzwerkzugriffe. `game.json` ist die einzige Spieldatenquelle; dadurch können spätere Texte mit `</script>`, HTML oder Unicode nicht in ein Script-Element ausbrechen. Die UI schreibt Texte nur über `textContent`. Die Canvas-Szene rendert eigene geometrische Pixel-Platzhalter, Entitäten nach Klasse, Spieler, auswählbare Inventargegenstände, Hotspotauswahl, kontextuelle Aktionen, Portalzustand sowie umschaltbare Walkbox- und Kollisionsdaten.

Der generische JavaScript-Core (`runtime-core.js`, für Node bytegleich als `.cjs`) ist von DOM und Canvas getrennt. Python und JavaScript interpretieren Bedingungen, Effekte, Verfügbarkeit, Container im Inventar, Endbedingungen, Polygonränder, Projektion und numerische Tie-Breaks gleich. Automatische Node-Replays führen für Labor und Piratenhafen exakt den Python-Solverweg aus und vergleichen nach jedem Schritt Inventar, Objekte, Ziele, Flags und Abschluss. Golden-Room-IDs steuern keine Sonderzweige. Da `game.json` per `fetch` geladen wird, startet der Export über einen lokalen statischen HTTP-Server statt direkt per `file://`.

Generierte Dateien werden zunächst vollständig in ein temporäres Nachbarverzeichnis geschrieben, validiert und erst danach atomar an den Zielpfad verschoben. Einzelne JSON-, HTML-, CSS- und JavaScript-Dateien werden ebenfalls über temporäre Dateien ersetzt. Vorhandene Zielverzeichnisse werden nicht überschrieben; bei Validierungs- oder Solverfehlern bleibt kein Teilexport zurück.

## Spätere Entwicklung

Phase 2 implementiert dafür einen provider-neutralen OpenAI-kompatiblen Director mit zwingend explizitem Responses-v1- oder Chat-Completions-JSON-Schema-Protokoll, ohne Protokollfallback. Ein versioniertes, providerfreundliches WireSpec wird deterministisch in das unveränderte interne AdventureSpec übersetzt. Ein Schema-Preflight, injizierbarer HTTP-Transport mit harter Totalgrenze und genau ein begrenzter Reparaturversuch härten die Providergrenze. Die vollständige Vertrauensgrenze, Konfiguration und Provenienz beschreibt [`pixelworld-0.6.3-openai-compatible-director.md`](pixelworld-0.6.3-openai-compatible-director.md). Die erste externe Modellanfrage bleibt eine bewusst separate Freigabe; Phase 2 wurde ausschließlich mit synthetischen Fake-Antworten getestet. Danach können mehrere Räume, umfangreichere typisierte Prädikate, Dialogdaten, Autorenwerkzeuge und hochwertige eigene Assets folgen.

Aus validierten Specs, Compilerentscheidungen, Solverpfaden und Rasterisierungen lassen sich später synthetische Trainingspaare bilden. Ein trainiertes PixelWorld-Modell kann dann Layoutvorschläge, Walkboxen, Asset-Varianten oder visuelle Komposition vorhersagen. Referenzprüfung, Zustandsmaschine, Kollision, Solver, Sicherheitsgrenzen und Export bleiben deterministisch und außerhalb des Modells.
