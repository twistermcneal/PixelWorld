from dataclasses import asdict, dataclass, field
from typing import Any

from pixelworld.config import DEFAULT_EVALUATION_SEEDS


VERSION = "0.6.2"
STUDY_NAME = "0.6.2-placement"
SLOT_LATENT_DIM = 8
TERRAIN_LATENT_DIM = 7
LAYOUT_DIM = TERRAIN_LATENT_DIM + 8 * SLOT_LATENT_DIM
CONDITION_DIM = 4 + 6 + LAYOUT_DIM
LOCAL_OFFSET_PIXELS = 8
VARIANTS = ("A", "B", "C", "D", "E")
OFFSET_VARIANTS = ("C", "D", "E")
OFFSET_RADII = (8, 12, 16)
AUXILIARY_LOSS_WEIGHT = 0.25
OFFSET_LOSS_WEIGHT = 0.5


@dataclass
class PlacementConfig:
    version: str = VERSION
    variant: str = "B"
    samples: int = 14_000
    batch_size: int = 128
    epochs: int = 45
    seed: int = 42
    learning_rate: float = 5e-4
    optimizer: str = "AdamW"
    offset_radius: int = LOCAL_OFFSET_PIXELS
    offset_loss_weight: float = OFFSET_LOSS_WEIGHT
    auxiliary_loss_weight: float = AUXILIARY_LOSS_WEIGHT
    num_workers: int = 0
    evaluation_seeds: tuple[int, ...] = field(default_factory=lambda: DEFAULT_EVALUATION_SEEDS)

    def validate(self):
        if self.version != VERSION:
            raise ValueError(f"Unsupported study version: {self.version!r}")
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown placement variant: {self.variant!r}")
        if self.variant == "A":
            if self.offset_loss_weight != OFFSET_LOSS_WEIGHT:
                raise ValueError("Variant A does not permit changed loss weights")
        if self.offset_radius != LOCAL_OFFSET_PIXELS:
            raise ValueError(
                f"explicit 0.6.2 latent offsets require offset_radius={LOCAL_OFFSET_PIXELS}"
            )
        for name in ("samples", "batch_size", "epochs"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0 or self.optimizer != "AdamW":
            raise ValueError("PixelWorld 0.6.2 requires AdamW and a positive learning rate")
        if self.num_workers != 0:
            raise ValueError("PixelWorld 0.6.2 requires num_workers=0")
        if not self.evaluation_seeds:
            raise ValueError("evaluation_seeds must not be empty")
        return self

    @property
    def uses_stable_anchors(self):
        return self.variant != "A"

    @property
    def uses_offset(self):
        return self.variant in OFFSET_VARIANTS

    @property
    def detach_offset_input(self):
        return self.variant in ("C", "E")

    @property
    def uses_auxiliary_xy(self):
        return self.variant == "E"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evaluation_seeds"] = list(self.evaluation_seeds)
        data["baseline_version"] = "0.6.1"
        data["slot_latent_dim"] = SLOT_LATENT_DIM
        data["layout_dim"] = LAYOUT_DIM
        data["condition_dim"] = CONDITION_DIM
        data["offset_target"] = "raw_seed_latent"
        data["anchor_frame"] = "resolved_region_candidate_bbox"
        data["seeds"] = {
            "python": self.seed,
            "numpy": self.seed,
            "pytorch": self.seed,
            "dataloader": self.seed,
            "evaluation": list(self.evaluation_seeds),
        }
        return data

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        values.pop("baseline_version", None)
        values.pop("seeds", None)
        for name in ("slot_latent_dim", "layout_dim", "condition_dim", "offset_target", "anchor_frame"):
            values.pop(name, None)
        values["evaluation_seeds"] = tuple(values["evaluation_seeds"])
        return cls(**values).validate()


def variant_spec(variant: str) -> dict[str, Any]:
    config = PlacementConfig(variant=variant).validate()
    return {
        "variant": variant,
        "stable_anchors": config.uses_stable_anchors,
        "offset": config.uses_offset,
        "offset_input_detached": config.detach_offset_input,
        "auxiliary_xy": config.uses_auxiliary_xy,
        "offset_loss_weight": config.offset_loss_weight if config.uses_offset else 0.0,
        "auxiliary_loss_weight": config.auxiliary_loss_weight if config.uses_auxiliary_xy else 0.0,
    }
