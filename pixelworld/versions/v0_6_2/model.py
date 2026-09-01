import torch
from torch import nn

from pixelworld.config import COORD_CLASSES
from pixelworld.model import LandscapeNet

from .config import CONDITION_DIM, OFFSET_VARIANTS, VARIANTS


class LandscapeNet062(LandscapeNet):
    def __init__(self, variant, condition_dim=CONDITION_DIM, hidden=320, detach_placement_queries=False):
        if variant not in OFFSET_VARIANTS:
            raise ValueError(f"LandscapeNet062 requires an offset variant, got {variant!r}")
        super().__init__(condition_dim=condition_dim, hidden=hidden)
        self.variant = variant
        self.detach_placement_queries = bool(detach_placement_queries)
        self.offset_head = nn.Linear(hidden, 2)
        if variant == "E":
            self.xy_auxiliary_head = nn.Linear(hidden, 2)

    def forward(self, x):
        terrain = self.terrain_encoder(x)
        placement_queries = (
            self.slot_queries.weight.detach()
            if self.detach_placement_queries
            else self.slot_queries.weight
        )
        placement = self.slots_for(
            self.placement_encoder(x), self.placement_decoder, queries=placement_queries
        )
        attributes = self.slots_for(self.attribute_encoder(x), self.attribute_decoder)
        offset_input = placement.detach() if self.variant in ("C", "E") else placement
        outputs = (
            self.terrain_numeric(terrain).reshape(-1, 5, COORD_CLASSES),
            self.orientation_head(terrain),
            self.biome_head(terrain),
            self.region_head(placement),
            self.anchor_head(placement),
            self.presence_head(self.presence_encoder(x)),
            self.class_head(attributes),
            self.action_head(attributes),
            self.trigger_head(attributes),
            torch.tanh(self.offset_head(offset_input)),
        )
        if self.variant == "E":
            outputs += (torch.sigmoid(self.xy_auxiliary_head(placement)),)
        return outputs


def create_model(variant, detach_placement_queries=False):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown placement variant: {variant!r}")
    if variant == "A":
        return LandscapeNet()
    if variant == "B":
        return LandscapeNet(condition_dim=CONDITION_DIM)
    return LandscapeNet062(variant, detach_placement_queries=detach_placement_queries)
