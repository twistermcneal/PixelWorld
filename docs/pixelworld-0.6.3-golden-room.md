# PixelWorld 0.6.3: Golden Room

## Raum

**Professor Knallberts chronochemisches Labor** ist ein farbenfrohes, dunkles Science-Fiction-Labor in 128 × 72 logischen Pixeln. Eine leuchtende Zeitmaschine besetzt die Mitte. Links stehen rote und blaue Chemikalien, rechts Mischflasche, grüner Katalysator und Professor Knallbert; ein Zeitportal bildet den Ausgang. Roboterarm, Zahnräder, Neonfarben, Funkenstimmung und große Vordergrundformen erzeugen die Tiefenstaffelung.

Die Browserdarstellung verwendet klar erkennbare eigene geometrische Platzhalter. Es wurden keine externen oder markengebundenen Assets übernommen.

## Golden-AdventureSpec

Die Spec enthält einen Ort, neun Raumobjekte, einen NPC und vier Inventardefinitionen. Pflichtobjekte sind Zeitmaschine, Bedienpult, rotes und blaues Reagenz, Mischflasche und Portal. Der grüne Katalysator sowie Laborrequisiten sind bewusst optional. Das Theme `mad_scientist_lab` validiert alle verwendeten Klassen.

Initialzustand:

```text
player_position = [15, 60]
inventory = []
coolant_red.taken = false
coolant_blue.taken = false
mixing_flask.taken = false
mixing_flask.contents = "empty"
time_machine.cooled = false
time_portal.active = false
cool_time_machine.completed = false
```

## Navigation und Walkboxes

Drei konvexe Bodenpolygone bilden linken, vorderen und rechten Laufbereich. Der vordere Bereich verbindet die Seiten vor der zentralen Maschinenkollision. Linke und rechte Labortische sind ebenfalls Kollisionen. Alle Pflicht-Hotspots besitzen einen begehbaren Walk-to-Punkt; die Vordergrundflaschen und -rohre sind Okklusionspolygone mit eigener Z-Ebene.

Der Validator erreicht vom Start aus alle sieben Pflichtentitäten einschließlich Professor und das Zeitportal, ohne ein Kollisionspolygon zu durchqueren. Klicks außerhalb der Walkboxes werden auf den nächsten gültigen Randpunkt projiziert.

## Rätsel und kürzeste Lösung

Die ersten drei Schritte sind unabhängig und können in beliebiger Reihenfolge ausgeführt werden. Die kanonische Solverreihenfolge ist:

1. `take_blue` – blaues Kühlreagenz nehmen
2. `take_flask` – Mischflasche nehmen
3. `take_red` – rotes Kühlreagenz nehmen
4. `mix_coolant` – beide Reagenzien in der Flasche kombinieren
5. `cool_machine` – das fertige Chronokühlmittel mit der Zeitmaschine benutzen

Der letzte Schritt setzt `time_machine.cooled`, `time_portal.active` und `cool_time_machine.completed` auf `true`. Damit ist die Endbedingung erfüllt. Verbrauchte Komponenten werden deterministisch entfernt; es gibt keine Sackgasse.

## CLI und Artefakte

```powershell
python -m pixelworld.cli adventure-generate --version 0.6.3 --director fixture --prompt "Ein verrückter Wissenschaftler repariert seine Zeitmaschine" --output outputs/adventures/0.6.3-golden-lab
python -m pixelworld.cli adventure-validate --spec outputs/adventures/0.6.3-golden-lab/adventure_spec.json
python -m pixelworld.cli adventure-solve --game outputs/adventures/0.6.3-golden-lab/game.json
python -m http.server 8000 --directory outputs/adventures/0.6.3-golden-lab
```

Der Ausgabeordner enthält AdventureSpec, RoomSpec, Scene Graph, vollständiges Spiel, Validierungsbericht, Lösung, HTML, JavaScript, CSS und `assets/`. Er ist durch `.gitignore` ausgeschlossen und gehört nicht in Commits.

## Aktuelle Platzhalter und nächste Schritte

- Figuren und Objekte sind farbcodierte Canvas-Platzhalter statt finaler Sprite-Sheets.
- Bewegung nutzt eine einfache interpolierte Figur ohne Laufzyklen.
- Professor Knallbert besitzt eine feste Hinweiszeile statt Dialogbaum.
- Das Portal beendet Phase 1 als Zustandsziel; ein Folgeraum existiert noch nicht.
- Semantische Audio-, Animations- und Assetreferenzen benötigen später eigene versionierte Verträge.

