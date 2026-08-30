import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_0_6_1_notebook.py"
NOTEBOOK = ROOT / "notebooks" / "PixelWorld_0_6_1.ipynb"


def notebook_hash():
    return hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()


def test_notebook_builder_is_utf8_idempotent_and_uses_core():
    subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, check=True)
    first = notebook_hash()
    subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, check=True)
    assert notebook_hash() == first

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "PixelWorld 0.6.1 —" in sources
    assert "from pixelworld" in sources
    for duplicate in (
        "class LandscapeNet",
        "def generate_landscape",
        "def compute_losses",
        "def evaluate_model",
        "exec(",
    ):
        assert duplicate not in sources
