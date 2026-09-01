import numpy as np
import torch
from torch import nn

from .config import BIOMES, COORD_CLASSES, LAYOUT_DIM, MAX_SLOTS
from .generation import layout_from_seed


def prompt_vector(prompt):
    p = prompt.lower()
    v = np.zeros(len(BIOMES) + 6, np.float32)
    for i, biome in enumerate(BIOMES):
        v[i] = float(biome in p)
    for i, word in enumerate(["coast", "beach", "forest", "rock", "portal", "dark"]):
        v[len(BIOMES) + i] = float(word in p)
    return v


def condition_vector(prompt, seed):
    return np.concatenate([prompt_vector(prompt), layout_from_seed(seed)]).astype(np.float32)


class LandscapeNet(nn.Module):
    def __init__(self, condition_dim=len(BIOMES) + 6 + LAYOUT_DIM, hidden=320):
        super().__init__()
        self.slots = MAX_SLOTS

        def enc():
            return nn.Sequential(
                nn.Linear(condition_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )

        def dec():
            return nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )

        self.terrain_encoder = enc()
        self.placement_encoder = enc()
        self.presence_encoder = enc()
        self.attribute_encoder = enc()
        self.terrain_numeric = nn.Linear(hidden, 5 * COORD_CLASSES)
        self.orientation_head = nn.Linear(hidden, 4)
        self.biome_head = nn.Linear(hidden, 4)
        self.slot_queries = nn.Embedding(MAX_SLOTS, hidden)
        self.placement_decoder = dec()
        self.attribute_decoder = dec()
        self.region_head = nn.Linear(hidden, 4)
        self.anchor_head = nn.Linear(hidden, 16)
        self.presence_head = nn.Linear(hidden, MAX_SLOTS)
        self.class_head = nn.Linear(hidden, 4)
        self.action_head = nn.Linear(hidden, 3)
        self.trigger_head = nn.Linear(hidden, 4)

    def slots_for(self, world, decoder, queries=None):
        query_values = self.slot_queries.weight if queries is None else queries
        q = query_values[None].expand(world.shape[0], -1, -1)
        c = world[:, None, :].expand(-1, self.slots, -1)
        return decoder(torch.cat([c, q], -1))

    def forward(self, x):
        t = self.terrain_encoder(x)
        p = self.slots_for(self.placement_encoder(x), self.placement_decoder)
        a = self.slots_for(self.attribute_encoder(x), self.attribute_decoder)
        return (
            self.terrain_numeric(t).reshape(-1, 5, COORD_CLASSES),
            self.orientation_head(t),
            self.biome_head(t),
            self.region_head(p),
            self.anchor_head(p),
            self.presence_head(self.presence_encoder(x)),
            self.class_head(a),
            self.action_head(a),
            self.trigger_head(a),
        )
