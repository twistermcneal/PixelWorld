import os
import sys
from pathlib import Path

from ipykernel.kernelspec import install as install_kernel
from nbclient import NotebookClient
from nbformat import reads


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "PixelWorld_0_6_1.ipynb"
JUPYTER_DATA = ROOT / "outputs" / ".jupyter-smoke"
KERNEL_NAME = "pixelworld-smoke"


def main():
    JUPYTER_DATA.mkdir(parents=True, exist_ok=True)
    os.environ["JUPYTER_DATA_DIR"] = str(JUPYTER_DATA)
    os.environ["JUPYTER_CONFIG_DIR"] = str(JUPYTER_DATA / "config")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(JUPYTER_DATA / "runtime")
    os.environ["IPYTHONDIR"] = str(JUPYTER_DATA / "ipython")
    os.environ.setdefault("MPLBACKEND", "Agg")
    install_kernel(
        kernel_name=KERNEL_NAME,
        display_name="PixelWorld smoke",
        user=True,
        env={"PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"},
    )
    notebook = reads(NOTEBOOK.read_text(encoding="utf-8"), as_version=4)
    training_cell = notebook.cells[7]
    training_cell.source = training_cell.source.replace(
        "TRAINING_CONFIG = RunConfig()",
        "TRAINING_CONFIG = RunConfig(samples=8, batch_size=4, epochs=1, evaluation_seeds=(500000,))",
    )
    NotebookClient(
        notebook,
        timeout=300,
        kernel_name=KERNEL_NAME,
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    print(
        f"Notebook smoke passed with {len(notebook.cells)} cells on {sys.executable}",
        flush=True,
    )


if __name__ == "__main__":
    main()
