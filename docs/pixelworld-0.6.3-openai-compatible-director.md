# PixelWorld 0.6.3: OpenAI-kompatibler Story Director

## Vertrauensgrenze

Der Story Director ist eine nicht vertrauenswürdige Datenquelle. Er liefert ausschließlich Daten. Compiler, internes `AdventureSpec`, Theme-Templates, Validator, Solver sowie Python- und JavaScript-Runtime bleiben unverändert die Autorität. Modelltext wird nie als Code, Ausdruck oder Template ausgeführt.

```text
Provider-WireSpec v1
→ striktes Einzelobjekt-JSON
→ deterministische WireSpec-zu-AdventureSpec-Transformation
→ unveränderte AdventureSpec-Validierung
→ Compiler → Game-Validator → Solver
→ atomarer Export
```

## Explizite Konfiguration

Es gibt keine Protokollerkennung und keinen Fallback. Neben Base-URL, API-Key und Modell muss genau ein Protokoll gesetzt werden:

```powershell
$env:PIXELWORLD_LLM_BASE_URL = "http://gx10.example:8000/v1"
$env:PIXELWORLD_LLM_API_KEY = "..."
$env:PIXELWORLD_LLM_MODEL = "explicit-model-id"
$env:PIXELWORLD_LLM_PROTOCOL = "chat-completions-json-schema"
```

Erlaubt sind:

- `responses-v1`: `POST <BASE_URL>/responses`
- `chat-completions-json-schema`: `POST <BASE_URL>/chat/completions`

Dieselben Werte lassen sich über `--llm-base-url`, `--llm-api-key`, `--llm-model` und `--llm-protocol` setzen. Für den Schlüssel ist die Umgebungsvariable vorzuziehen. Redirects werden nicht verfolgt; URL, Antwortgröße sowie Connect-, Read- und harter Gesamttimeout sind begrenzt.

Die Laufzeitparameter folgen strikt `CLI > Umgebungsvariable > sicherer Default`:

| Bedeutung | Umgebungsvariable | CLI | Default | Maximum |
|---|---|---|---:|---:|
| Connect-Timeout | `PIXELWORLD_LLM_CONNECT_TIMEOUT` | `--llm-connect-timeout` | 5 s | 30 s |
| Read-Timeout | `PIXELWORLD_LLM_READ_TIMEOUT` | `--llm-read-timeout` | 20 s | 600 s |
| harter Total-Timeout | `PIXELWORLD_LLM_TOTAL_TIMEOUT` | `--llm-total-timeout` | 30 s | 900 s |
| Ausgabetokens | `PIXELWORLD_LLM_MAX_OUTPUT_TOKENS` | `--llm-max-output-tokens` | 12.000 | 20.000 |
| HTTP-Antwortgröße | `PIXELWORLD_LLM_MAX_RESPONSE_BYTES` | `--llm-max-response-bytes` | 524.288 Bytes | 524.288 Bytes |

Timeouts müssen endlich und positiv sein; Booleanwerte sind keine Zahlen. Connect und Read dürfen den Total-Timeout nicht überschreiten. Token- und Bytelimits müssen positive Ganzzahlen sein. Null, Unendlich und unbeschränkte Werte werden abgewiesen. Preflight und Generierung verwenden denselben Resolver und dieselbe validierte effektive Konfiguration.

Für den ersten Smoke gegen den derzeitigen GX10-vLLM-Server ist `chat-completions-json-schema` zu verwenden. Dieser Pfad erwartet eine Chat-Template-fähige Textgeneration und extrahiert ausschließlich `choices[0].message.content`. Aktuelle vLLM-Versionen dokumentieren zwar zusätzlich eine Responses API, deren tatsächliche Verfügbarkeit und Structured-Output-Unterstützung hängt aber von der installierten Serverversion und dem Modell ab. `responses-v1` darf deshalb auf dem GX10 erst nach einem erfolgreichen expliziten Preflight gewählt werden. Die aktuelle offizielle vLLM-Dokumentation führt beide Endpunkte auf: [OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/).

## Protokollverträge

`responses-v1` folgt dem offiziellen [OpenAI Responses API](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)-Vertrag: Systemgrenze in `instructions`, Prämisse in `input`, Schema in `text.format` mit `type: "json_schema"`, außerdem `store: false` und `stream: false`. Akzeptiert wird entweder ein alleinstehendes nichtleeres `output_text` oder exakt ein `output[].content[]`-Element vom Typ `output_text`; mehrere Outputtexte werden abgewiesen.

`chat-completions-json-schema` sendet exakt:

```json
{
  "model": "<explizite Modell-ID>",
  "messages": [
    {"role": "system", "content": "<fester Story-Director-Prompt>"},
    {"role": "user", "content": "<Prämisse oder Repair-Daten>"}
  ],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "pixelworld_adventure_spec_0_6_3",
      "strict": true,
      "schema": "<Provider-WireSpec-v1-Schema>"
    }
  },
  "max_tokens": 12000,
  "stream": false
}
```

Der Response-Envelope muss genau eine Choice mit nichtleerem stringförmigem `message.content` enthalten. Mehrere Choices, Refusals ohne Content und falsche Envelopes werden abgewiesen. Repair-Requests bleiben immer auf dem konfigurierten Protokoll.

## Internes Schema und Provider-WireSpec

`adventure_spec_json_schema()` beschreibt weiterhin das vollständige interne Datenmodell. Es enthält absichtlich Ausdrucksmittel wie `propertyNames`, Tupel über `prefixItems`, schemawertige `additionalProperties` und skalare `anyOf`-Zweige. Diese sind für die interne Validierung richtig, aber nicht als gemeinsamer Strict-Structured-Output-Subset aller Provider zuverlässig.

Das gesendete, versionierte `pixelworld-adventure-wire-1` verwendet daher einen konservativen gemeinsamen Subset. `validate_provider_schema()` prüft ihn vor jedem Request programmgesteuert. Zugelassen sind nur Objekte mit vollständigem `required`, `additionalProperties: false`, begrenzte homogene Arrays, primitive Typen, `enum`, `const`, String- und Zahlenlimits. `propertyNames`, `prefixItems`, `anyOf` und schemawertige zusätzliche Properties kommen nicht vor.

Dynamische Zustände werden als begrenzte Liste eindeutiger Namen übertragen. Skalare Werte haben eine diskriminierte, aber unionsfreie Form:

```json
{
  "name": "taken",
  "type": "boolean",
  "boolean_value": false,
  "string_value": "",
  "integer_value": 0,
  "number_value": 0.0
}
```

Feste interne Werte werden auf dem Wire nicht kreativ generiert: `schema_version` wird aus der Wire-Version abgeleitet, Raumgröße `128×72` ergänzt, Spielerposition als `{x,y}` übertragen und ein leeres Portalziel deterministisch zu `null` transformiert. Round-trip-Tests beweisen für beide synthetischen Abenteuer die identische Rückgewinnung des internen `AdventureSpec`. Runtime-Semantik ändert sich dadurch nicht.

## Schema-Preflight

Der Check erzeugt keinen Spielordner und keine Adventure-Ausgabe:

```powershell
python -m pixelworld.cli adventure-director-check `
  --version 0.6.3 `
  --llm-protocol chat-completions-json-schema
```

Er berichtet getrennt über gültige Base-URL, explizites Modell, Erreichbarkeit, Modellfund in `/models` sofern unterstützt, Existenz des gewählten Protokollendpunkts und Annahme des tatsächlich verwendeten WireSpec-Schemas. Der POST fordert lediglich eine feste, minimale, begrenzte Schema-Probe an; er kompiliert kein Spiel und schreibt nichts nach `outputs/`. HTTP 404 und 400 bleiben unterscheidbar. Der Check erkennt oder wechselt das Protokoll nie automatisch.

## Repair, Limits und Provenienz

Nach Decode-, Transformations-, Spec-, Compiler-, Validator- oder Solverfehler gibt es genau einen Repair-Request mit der vorherigen Rohantwort und höchstens acht bereinigten Fehlern zu je 240 Zeichen. Transportfehler, Redirects, Timeouts und Größenfehler werden nicht wiederholt. Nach zwei ungültigen Antworten bleibt der Zielordner aus.

Der Transport verwendet keine privaten `urllib`-Attribute. `http.client` und öffentliche Socket-Timeouts trennen Connect und Read; eine äußere Deadline erzwingt zusätzlich den harten Total-Timeout auch bei langsam tröpfelnden Antworten.

Nur erfolgreiche Modellläufe schreiben `director_provenance.json`. Gespeichert werden unter anderem das tatsächlich gewählte Protokoll (`responses-v1` oder `chat-completions-json-schema`), Modell, sanitierte Base-URL, die fünf effektiven Laufzeitparameter, Prompt-/Response-Hashes, Versuchszahl, Compile-Digest, Zeit, Python- und Git-Identität. API-Key, Authorization-Header, Rohantwort, Reasoning und Response-Body eines Fehlers werden nie persistiert oder in Exceptions übernommen.

## Offline-Tests

Alle Providerpfade werden mit injizierten Fake-Transporten geprüft. Lokale HTTP-Testserver decken Read-Timeout, harten Total-Timeout unter Tröpfelantworten, Redirect und Größenlimit ab. Beide synthetischen Abenteuer laufen weiterhin durch Compiler, Validator, Solver, Python-Replay, schrittweisen Node-Paritätsvergleich und Browserexport. In dieser Phase wurde kein GX10- oder sonstiger externer Modellaufruf durchgeführt.
