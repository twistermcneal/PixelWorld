# PixelWorld 0.6.3 Phase 2: OpenAI-kompatibler Story Director

## Vertrauensgrenze

Der Story Director ist eine nicht vertrauenswürdige Datenquelle. Er darf ausschließlich eine `AdventureSpec` vorschlagen. Compiler, Theme-Template, State-Schema, Geometrievalidator, Solver, Python-Runtime und JavaScript-Runtime bleiben die Autorität. Modelltext wird niemals als Python, JavaScript, Shell, Template oder Ausdruck ausgeführt.

Die Pipeline akzeptiert eine Modellantwort erst nach diesen Gates:

```text
striktes Einzelobjekt-JSON
→ validate_adventure_spec
→ compile_adventure
→ validate_game
→ begrenzter State-Space-Solver
→ atomarer Export
```

Ein sprachlich überzeugendes Modellresultat ist kein Ersatz für Validierung und Solver: Nur diese beweisen referenzielle Integrität, unterstützte Klassen/Zonen, typisierte Zustände, erreichbare Hotspots und einen tatsächlich ausführbaren Lösungsweg.

## Konfiguration

Der Director wählt keinen Provider automatisch. Er verwendet explizit das OpenAI-kompatible Responses-v1-Protokoll und `POST <BASE_URL>/responses`. `BASE_URL` ist der API-Root, beispielsweise `https://api.example.test/v1`, nicht bereits der `/responses`-Endpunkt.

Konfiguration erfolgt per CLI oder Umgebungsvariablen:

```powershell
$env:PIXELWORLD_LLM_BASE_URL = "https://api.example.test/v1"
$env:PIXELWORLD_LLM_API_KEY = "..."
$env:PIXELWORLD_LLM_MODEL = "explicit-model-id"

python -m pixelworld.cli adventure-generate `
  --version 0.6.3 `
  --director openai-compatible `
  --prompt "Ein tollpatschiger Erfinder muss vor Mitternacht ein Zeitportal reparieren" `
  --output outputs/adventures/my-game
```

Alternativ existieren `--llm-base-url`, `--llm-api-key` und `--llm-model`. Für den Schlüssel ist die Umgebungsvariable vorzuziehen, damit er nicht in Shell-Historie oder Prozessargumenten erscheint. Die Base-URL muss `http` oder `https`, einen expliziten Host und keine Credentials, Query oder Fragment besitzen. Redirects werden nicht verfolgt. Connect-, Read- und Gesamttimeout sowie Antwortgrößen sind begrenzt.

Phase 2 erlaubt nur die bereits kompilierbaren Themes:

- `mad_scientist_lab`
- `pirate_harbor`

Die übrigen Ontologie-Themes bleiben gesperrt, bis ein deterministisches Layouttemplate existiert.

## Requestvertrag

Der Request entspricht der offiziellen OpenAI-Responses-Konvention für Structured Outputs:

```json
{
  "model": "<explizite Modell-ID>",
  "instructions": "<fester Story-Director-Systemprompt>",
  "input": [
    {
      "role": "user",
      "content": [{"type": "input_text", "text": "<Storyidee>"}]
    }
  ],
  "text": {
    "format": {
      "type": "json_schema",
      "name": "pixelworld_adventure_spec_0_6_3",
      "strict": true,
      "schema": "<vollständiges AdventureSpec-JSON-Schema>"
    }
  },
  "max_output_tokens": 12000,
  "store": false,
  "stream": false
}
```

Der feste Prompt erklärt, dass Benutzertext nur eine Prämisse ist. Er übergibt erlaubte Themes, Klassen, Rollen, Zonen, Verben, Operatoren und Limits, enthält aber kein vollständiges Golden-Fixture. Geschützte Figuren und kopierte Dialoge werden ausdrücklich ausgeschlossen.

Der Authorization-Header existiert nur im flüchtigen Transportrequest. Er wird weder geloggt noch gespeichert, gehasht oder in Fehlermeldungen aufgenommen.

## Responsevertrag

Der HTTP-Envelope muss JSON sein und entweder genau ein stringförmiges `output_text` oder genau einen Content-Eintrag mit `type: "output_text"` enthalten. Dieser Text muss nach optionalem äußerem Whitespace exakt ein JSON-Objekt sein.

Abgewiesen werden insbesondere Markdown-Fences, Prosa, Suffixtext, mehrere Dokumente, Arrays, NaN/Infinity, mehr als 128 KiB Modelltext, mehr als 20 JSON-Ebenen, mehr als 10.000 Knoten sowie jede Verletzung von Spec, Ontologie, Template, Compiler, Validator oder Solver. Es gibt keine Reparaturheuristik, JSON-Extraktion oder tolerante Dekodierung.

Die maschinenlesbare Schemafunktion liegt in `pixelworld/adventure/structured_schema.py`. Sie begrenzt Listen und Texte und weist unbekannte Felder über `additionalProperties: false` ab. Dynamische `initial_state`-Felder sind ausschließlich begrenzte skalare Deklarationen; Bedingungen und Effekte dürfen danach nur exakt diese Felder und Typen verwenden.

## Begrenzter Repair-Loop

Eine bereits gültige und lösbare erste Antwort erzeugt keinen zweiten Request. Bei einem Decode-, Spec-, Compiler-, Validator- oder Solverfehler folgt genau ein Reparaturrequest. Er enthält nur:

- die vorherige Modellantwort,
- höchstens acht bereinigte Fehler mit jeweils maximal 240 Zeichen,
- die Aufforderung, ein vollständiges korrigiertes JSON-Objekt zurückzugeben.

Transportfehler, Timeouts, Redirects und Größenüberschreitungen werden nicht durch einen zweiten Netzwerkversuch kaschiert. Scheitert auch die Reparatur, endet die Pipeline verständlich und ohne Zielordner. Stacktraces, lokale Pfade, Request-Header und Secrets werden nicht an das Modell gesendet.

## Provenienz

Nur ein erfolgreicher OpenAI-kompatibler Lauf erzeugt `director_provenance.json`:

- Schema-Version und Director-Typ
- `openai-responses-v1` als Providerprotokoll
- explizite Modell-ID
- sanitierte Base-URL
- SHA-256 des Benutzerprompts
- SHA-256 jedes rohen `output_text`
- Versuchszahl und begrenzter Validierungsstatus je Versuch
- Compile-Digest
- UTC-Zeitpunkt und Python-Version
- Git-Commit und Dirty-Status, wenn zuverlässig ermittelbar

API-Key, Authorization-Header, vollständige Header, URLs mit Credentials, Modell-Reasoning und Chain-of-Thought werden nicht persistiert. Ungültige Antworten werden standardmäßig überhaupt nicht unter `outputs/` gespeichert.

## Offline-Teststrategie

Automatisierte Tests verwenden ausschließlich einen injizierten `FakeTransport`. Sie prüfen Erfolg, Reparatur, endgültigen Abbruch, striktes JSON, Theme-/Template-Grenzen, Unlösbarkeit, Timeout, HTTP-Fehler, Redirect, Antwortgröße, Konfiguration und Secret-Leaks. Fixture- und JSON-Director werden zusätzlich unter einem verbotenen HTTP-Transport ausgeführt, um ihre Netzwerkfreiheit zu beweisen.

Zwei neue synthetische Modellantworten – Lyras Mitternachtswerkstatt und der Sturmpier – durchlaufen Schema, Compiler, Validator, Solver, Python-Replay, schrittweisen Node-Paritätsvergleich und Browserexport. In Entwicklung und Abschlussprüfung wurde kein echter externer API-Aufruf ausgeführt.

Die Requestform orientiert sich an der offiziellen [Responses API](https://developers.openai.com/api/reference/cli/resources/responses/methods/create) und deren [Structured-Outputs-Format](https://platform.openai.com/docs/api-reference/responses-streaming/response/web_search_call?lang=curl).
