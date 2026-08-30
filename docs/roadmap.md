# PixelWorld Roadmap

## Leitbild

PixelWorld erzeugt keine isolierten Bilder, sondern verschachtelte, reproduzierbare Welten. Jede räumliche Ebene besitzt einen Scene Graph, einen deterministischen Renderer und synchronisierte Logik-Maps.

```text
Landschaft → Siedlung → Bezirk → Gebäude → Innenraum → Objektwelt
```

## 0.6 – Meilenstein 3: Landschaften und Terrain

Status: in Auswertung.

- Biome und Küstenrichtungen
- Wasser, Sand, Gras, Erde, Fels und Schnee
- Uferlinie, Strandbreite und Felsigkeit
- Walkability-, Object- und Interaction-Maps
- Außenwelt-Slots für Baum, Fels, NPC und Portal

## 0.6.1 – Terrainregionen und Vegetation

Status: implementiert, Training und Auswertung offen.

- Positionen relativ zu Terrainregionen statt absoluter Pixelpositionen
- Regionen `BEACH`, `FOREST`, `OPEN_LAND`, `ROCK` und `SNOW`
- 16 kanonische Anchors pro Landmark Slot
- deterministischer Scatter-Layer für Bäume
- Dichte, Art, Mindestabstand und Vegetations-Seed
- Wege und freie Korridore von Vegetation ausnehmen

## 0.7 – Settlement Layer: Dörfer

Ein Dorf wird als eigener Scene Graph modelliert:

```json
{
  "type": "village",
  "center": [38, 27],
  "radius": 14,
  "style": "medieval",
  "layout": "organic",
  "population": 32,
  "entrances": ["south", "east"]
}
```

Der Dorf-Renderer erzeugt daraus Zentrum, Hauptwege, Zonen und bebaubare Grundstücke. Terrainregeln verhindern Siedlungen im Wasser oder an ungeeigneten Steilhängen.

## 0.7.1 – Straßen, Grundstücke und Gebäude

- Straßennetze `organic`, `grid` und `radial`
- Haupt- und Nebenwege
- Grundstücksparzellen entlang der Straßen
- Gebäudearten wie Wohnhaus, Gasthaus, Schmiede, Markt und Rathaus
- Eingänge an erreichbaren Straßenseiten
- deterministische Dekoration mit Zäunen, Brunnen, Laternen und Kisten

## 0.7.2 – Städte und Bezirke

Größere Städte erhalten District Slots:

- Altstadt
- Wohngebiet
- Markt
- Hafen
- Handwerk oder Industrie
- Verwaltung
- Stadtmauer und Tore

Jeder Bezirk beschreibt Zentrum, Ausdehnung, Dichte, Stil und Verbindungen zu benachbarten Bezirken. Ein Stadt-Renderer erzeugt daraus das übergeordnete Straßennetz und delegiert Grundstücke an den Building Layer.

## 0.8 – Durchgängige Welthierarchie

- Landschaft anklicken → Siedlung betreten
- Gebäude anklicken → Innenraum erzeugen
- Tür oder Portal anklicken → nächsten Scene Graph laden
- stabile IDs und Seeds auf allen Ebenen
- persistenter Story-State
- Änderungen bleiben bei späteren Besuchen erhalten

## Spätere Arbeit

- permutation-invariantes Slot-Matching
- freie Pixelmasken und Sprite-IDs
- Beziehungen wie `on`, `inside`, `locked_by` und `leads_to`
- Titelbildschirme als interaktive Welten
- Animation States und Sprite-Renderer
- Editor als kuratierte Datenquelle
