import os
from pathlib import Path

import pytest

from pixelworld.golden import compare_run_to_oracle


@pytest.mark.golden
def test_seed42_golden_run():
    run = os.environ.get("PIXELWORLD_GOLDEN_RUN")
    if not run:
        pytest.skip("set PIXELWORLD_GOLDEN_RUN to a completed full run directory")
    root = Path(__file__).resolve().parents[1]
    result = compare_run_to_oracle(Path(run), root / "outputs" / "0.6.1-reference")
    assert result["maximum_loss_deviation"] == 0.0
    assert result["maximum_metric_deviation"] == 0.0
    assert result["model_state_dict_bit_exact"]
