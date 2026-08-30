import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "PixelWorld_0_6.ipynb"
OUTPUT = ROOT / "notebooks" / "PixelWorld_0_6_1.ipynb"


def set_cell(notebook, index, text):
    cell = notebook["cells"][index]
    cell["source"] = text.splitlines(keepends=True)
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []


def build_notebook():
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))

    set_cell(
        notebook,
        0,
        """# PixelWorld 0.6.1 — Terrainregionen und deterministische Vegetation

Dieses Notebook ist die Forschungs- und Visualisierungsoberfläche für PixelWorld 0.6.1. Weltgenerierung, Placement, Modell, Training, Evaluation und Inferenz stammen aus dem gemeinsamen `pixelworld`-Core-Paket; dadurch bleibt die CLI numerisch identisch zum Notebook.
""",
    )
    set_cell(
        notebook,
        1,
        """# Das Repository-Root muss das aktuelle Arbeitsverzeichnis sein.
import numpy as np
import matplotlib.pyplot as plt

from pixelworld.config import DEFAULT_PROMPT, REGIONS, RunConfig, SIZE, TERRAINS
from pixelworld.evaluation import evaluate_model, vegetation_round_trip
from pixelworld.generation import REGION_PALETTE, generate_landscape
from pixelworld.inference import predict
from pixelworld.training import initialize_training, resolve_device, train_one_epoch
""",
    )
    set_cell(
        notebook,
        2,
        """## 1. Terrainregionen und Scatter-Layer

Der Terrain-Graph enthält Waldstufe und Vegetationsdichte. Die Region-Map unterscheidet Strand, offenes Land, Felsfeld und Wald. Landmark-Slots sagen Region und einen von 16 kanonischen Anchors voraus. Die Semantik ist im Core zentral implementiert.
""",
    )
    set_cell(
        notebook,
        3,
        """sample = generate_landscape(DEFAULT_PROMPT, 424242)
print("Seed:", sample.seed)
print("Terrain-Parameter:", sample.terrain_params)
print("Landmarks:", sample.objects)
print("Vegetation:", int(sample.vegetation.sum()), "Bäume")
""",
    )
    set_cell(
        notebook,
        4,
        """def show_landscape(world):
    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    items = [
        (world.rgb, "RGB"),
        (world.terrain, "Terrain"),
        (REGION_PALETTE[world.regions], "Regions"),
        (world.vegetation, "Vegetation"),
        (world.object_map, "Landmarks"),
        (world.interaction, "Interaction"),
    ]
    for axis, (data, title) in zip(axes, items):
        axis.imshow(data, interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    plt.tight_layout()
    plt.show()

show_landscape(sample)
""",
    )
    set_cell(
        notebook,
        5,
        """## 2. Getrennte Terrain-, Placement-, Presence- und Attribute-Pfade

Die unveränderte 0.6.1-Architektur, Initialisierungsreihenfolge, Targets und Loss-Gewichte leben in `pixelworld.model` und `pixelworld.training`. Die Standardkonfiguration entspricht dem Seed-42-Referenzlauf.
""",
    )
    set_cell(
        notebook,
        6,
        """DEVICE = resolve_device()
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    import torch
    torch.set_float32_matmul_precision("high")
    print("GPU:", torch.cuda.get_device_name(0))
""",
    )
    set_cell(
        notebook,
        7,
        """TRAINING_CONFIG = RunConfig()
training = initialize_training(TRAINING_CONFIG, DEVICE)
model = training.model
history = []

for epoch in range(TRAINING_CONFIG.epochs):
    record = {"epoch": epoch + 1, **train_one_epoch(training, DEVICE)}
    history.append(record)
    print(
        f"Epoch {epoch + 1:02d}: loss={record['loss']:.3f} "
        f"terrain={record['terrain_loss']:.3f} placement={record['placement_loss']:.3f} "
        f"presence={record['presence_loss']:.3f} class={record['class_loss']:.3f} "
        f"action={record['action_loss']:.3f} trigger={record['trigger_loss']:.3f}"
    )
""",
    )
    set_cell(
        notebook,
        8,
        """## 3. Auswertung über ungesehene Landschaften

Die gemeinsame Evaluation berechnet dieselben zwölf Metriken wie der archivierte 0.6.1-Referenzlauf. Inferenz und Rasterisierung werden ebenfalls aus dem Core importiert.
""",
    )
    set_cell(
        notebook,
        9,
        """metrics = evaluate_model(
    model,
    DEVICE,
    eval_seeds=TRAINING_CONFIG.evaluation_seeds,
    prompt=DEFAULT_PROMPT,
)
for name, value in metrics.items():
    print(f"{name:>16}: {value:.12f}")

single_prediction = predict(model, DEFAULT_PROMPT, 500000, DEVICE)
print("Einzelinferenz erfolgreich; Terrain-Parameter:", single_prediction[0].tolist())
print("Vegetations-Round-trip:", vegetation_round_trip())
""",
    )
    set_cell(
        notebook,
        10,
        """## Abgrenzung und nächste Schritte

Dieses Notebook bleibt die Forschungs- und Visualisierungsoberfläche für den gemeinsamen 0.6.1-Core. WebUI, Docker und PixelWorld 0.6.2 sind bewusst nicht Teil dieser Extraktion und folgen in separaten Schritten. Fachliche Schwächen der bestehenden Anchor- und Placement-Logik bleiben für die Golden-Parität unverändert.
""",
    )

    OUTPUT.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return OUTPUT


if __name__ == "__main__":
    print(build_notebook())
