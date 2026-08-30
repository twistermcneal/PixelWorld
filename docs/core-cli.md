# PixelWorld 0.6.1 Core und CLI

## Einrichtung unter Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -m "not golden"
```

`requirements.txt` enthält absichtlich keinen fest verdrahteten CUDA-Build. Für GPU-Training muss das zur lokalen NVIDIA-Hardware passende PyTorch-Wheel gemäß der PyTorch-Installationsauswahl installiert werden. Die CLI wählt CUDA automatisch, wenn `torch.cuda.is_available()` wahr ist, und verwendet sonst die CPU. `--device cpu` und `--device cuda` erlauben eine explizite Auswahl.

## Befehle

```powershell
python -m pixelworld.cli train --version 0.6.1
python -m pixelworld.cli train --version 0.6.1 --samples 2000 --batch-size 128 --epochs 3 --seed 42
python -m pixelworld.cli evaluate --run <run-id>
python -m pixelworld.cli infer --run <run-id> --prompt "tropical coast beach forest rock portal" --seed 500000
python -m pixelworld.cli runs
python -m pixelworld.cli resume --run <run-id>
```

Die Defaults für 0.6.1 sind 14.000 Samples, Batchgröße 128, 45 Epochen, Seed 42, Lernrate `5e-4`, AdamW, acht Slots und eine Weltgröße von 64 × 64. Ein Seed kontrolliert Python, NumPy, PyTorch und das DataLoader-Shuffle. Die 30 Evaluationsseeds bleiben fest.

## Run- und Recovery-Format

Jeder Lauf liegt unter `outputs/runs/<run-id>/` und enthält:

```text
config.json
status.json
training_history.csv
training_history.json
evaluation_metrics.json
run_summary.json
training.log
latest.pt
final.pt
```

`latest.pt` speichert Modell, Optimizer, abgeschlossene Epoche, Konfiguration, Zufallszustände und die bisherige Historie. `resume` rekonstruiert Dataset und DataLoader, lädt den Zustand und stellt die Zufallszustände vor der nächsten Epoche wieder her. Dadurch sind die Modell-Tensoren und Loss-Werte eines fortgesetzten Laufs identisch zu einem durchgehenden Lauf. Zeitmesswerte dürfen naturgemäß abweichen.

Checkpoint- und Statusdateien werden zunächst als temporäre Datei geschrieben und anschließend atomar ersetzt. `final.pt` ist mit `torch.load(..., weights_only=True)` ladbar. `outputs/`, Checkpoints und Experimente sind über `.gitignore` ausgeschlossen.

## Golden-Parität

Ein vollständiger Golden-Lauf wird explizit gestartet:

```powershell
python -m pixelworld.cli train --version 0.6.1 --samples 14000 --batch-size 128 --epochs 45 --seed 42 --run-id seed42-golden --device cuda
python -m pixelworld.cli golden --run seed42-golden
$env:PIXELWORLD_GOLDEN_RUN = (Resolve-Path outputs\runs\seed42-golden).Path
python -m pytest tests\test_golden_integration.py -m golden
```

Die Prüfung akzeptiert keine Toleranz: maximale Loss-Abweichung `0.0`, maximale Metrikabweichung `0.0` und ein bitgenau identischer `model_state_dict` sind erforderlich. Ein abweichender Hash der gesamten Checkpoint-Datei ist wegen Metadaten und Serialisierung zulässig. Das Oracle unter `outputs/0.6.1-reference` wird ausschließlich gelesen.

## Notebook

Der Builder liest und schreibt JSON explizit als UTF-8 und erzeugt bei wiederholtem Lauf dieselben Bytes:

```powershell
python scripts\build_0_6_1_notebook.py
python scripts\build_0_6_1_notebook.py
python scripts\smoke_0_6_1_notebook.py
```

Das Notebook bleibt eine Forschungs- und Visualisierungsoberfläche, importiert aber Generator, Modell, Training, Evaluation und Inferenz aus `pixelworld`. Produktivcode lädt kein Notebook und führt keinen Notebook-Code mit `exec()` aus.

WebUI, Docker und PixelWorld 0.6.2 gehören nicht zu dieser Implementierung und folgen in späteren Schritten. Insbesondere wurden Anchor-Logik, Modellköpfe, Losses und Gewichtungen nicht verändert.
