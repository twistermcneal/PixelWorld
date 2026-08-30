from dataclasses import asdict, dataclass, field
from typing import Any


VERSION = "0.6.1"
SEED = 42
SIZE = 64
MAX_SLOTS = 8
SLOT_LATENT_DIM = 6
TERRAIN_LATENT_DIM = 7
LAYOUT_DIM = TERRAIN_LATENT_DIM + MAX_SLOTS * SLOT_LATENT_DIM
COORD_CLASSES = SIZE + 1

BIOMES = ["temperate", "tropical", "arid", "tundra"]
ORIENTATIONS = ["north", "east", "south", "west"]
TERRAINS = {"water": 0, "sand": 1, "grass": 2, "dirt": 3, "rock": 4, "snow": 5}
REGIONS = ["beach", "open_land", "rock_field", "forest"]
LANDMARK_CLASSES = ["chest", "npc", "portal", "ruin"]
LANDMARK_SIZES = {"chest": (5, 4), "npc": (5, 9), "portal": (6, 8), "ruin": (8, 8)}
ACTIONS = ["LOOK", "USE", "SCAN"]
TRIGGER_TYPES = ["NONE", "WORLD", "STORY", "SECRET"]

DEFAULT_TRAINING_SAMPLES = 14_000
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 45
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_EVALUATION_SEEDS = tuple(500_000 + i * 7_919 for i in range(30))
DEFAULT_PROMPT = "tropical coast beach forest rock portal"


@dataclass
class RunConfig:
    version: str = VERSION
    samples: int = DEFAULT_TRAINING_SAMPLES
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    seed: int = SEED
    learning_rate: float = DEFAULT_LEARNING_RATE
    optimizer: str = "AdamW"
    world_size: int = SIZE
    max_slots: int = MAX_SLOTS
    num_workers: int = 0
    evaluation_seeds: tuple[int, ...] = field(default_factory=lambda: DEFAULT_EVALUATION_SEEDS)

    def validate(self) -> "RunConfig":
        if self.version != VERSION:
            raise ValueError(f"Unsupported PixelWorld version: {self.version!r}; expected {VERSION}")
        for name in ("samples", "batch_size", "epochs"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.optimizer != "AdamW":
            raise ValueError("PixelWorld 0.6.1 requires optimizer='AdamW'")
        if self.world_size != SIZE or self.max_slots != MAX_SLOTS:
            raise ValueError("PixelWorld 0.6.1 world_size and max_slots are immutable")
        if self.num_workers != 0:
            raise ValueError("PixelWorld 0.6.1 requires num_workers=0")
        if not self.evaluation_seeds:
            raise ValueError("evaluation_seeds must not be empty")
        return self

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evaluation_seeds"] = list(self.evaluation_seeds)
        data["seeds"] = {
            "python": self.seed,
            "numpy": self.seed,
            "pytorch": self.seed,
            "dataloader": self.seed,
            "evaluation": list(self.evaluation_seeds),
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        values = dict(data)
        values.pop("seeds", None)
        values["evaluation_seeds"] = tuple(values.get("evaluation_seeds", DEFAULT_EVALUATION_SEEDS))
        return cls(**values).validate()
