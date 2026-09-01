"""Small versioned visual-theme ontology used by the compiler."""

from __future__ import annotations

from copy import deepcopy


ONTOLOGY_VERSION = "0.6.3"


def _theme(terrain, architecture, objects, npcs, palette, lighting, portals, foreground, roles):
    return {
        "terrain_classes": terrain,
        "architecture_classes": architecture,
        "object_classes": objects,
        "npc_archetypes": npcs,
        "palette": palette,
        "lighting_moods": lighting,
        "portal_types": portals,
        "foreground_types": foreground,
        "hotspot_roles": roles,
    }


THEMES = {
    "mad_scientist_lab": _theme(
        ["laboratory_floor", "metal_platform"], ["industrial_wall", "chemical_shelf"],
        ["time_machine", "control_console", "chemical_bottle", "mixing_flask", "time_portal", "robot_arm", "gear"],
        ["mad_scientist"], ["#07111f", "#18304a", "#28e7ff", "#ff3b81", "#ffd33d", "#9cff57"],
        ["neon_sparks"], ["time_portal"], ["large_bottle", "pipework"],
        ["machine", "console", "ingredient", "container", "npc", "exit", "scenery"],
    ),
    "pirate_harbor": _theme(["dock", "water"], ["wooden_pier", "tavern"], ["brass_key", "locked_chest", "harbor_exit", "barrel", "ship", "rope"], ["pirate"], ["#14213d", "#8d5524", "#fca311"], ["sunset"], ["gangplank"], ["cargo"], ["item", "container", "exit", "npc", "scenery"]),
    "forest_ruin": _theme(["forest_floor", "stone"], ["ruin_wall", "arch"], ["tree", "altar", "vine"], ["guardian"], ["#132a13", "#31572c", "#90a955"], ["dappled"], ["stone_arch"], ["foliage"], ["exit", "npc", "scenery"]),
    "spaceship": _theme(["deck", "service_grate"], ["bulkhead", "airlock"], ["console", "reactor", "crate"], ["crew", "android"], ["#090b1a", "#3a86ff", "#ff006e"], ["emergency"], ["airlock"], ["cabling"], ["exit", "npc", "machine"]),
    "medieval_village": _theme(["road", "grass"], ["cottage", "market_stall"], ["well", "cart", "sign"], ["villager"], ["#432818", "#99582a", "#ffe6a7"], ["daylight"], ["village_gate"], ["fence"], ["exit", "npc", "scenery"]),
}


class ThemeOntology:
    version = ONTOLOGY_VERSION

    def get(self, theme: str) -> dict:
        if theme not in THEMES:
            raise ValueError(f"unknown visual theme {theme!r}; known themes: {', '.join(sorted(THEMES))}")
        return deepcopy(THEMES[theme])

    def validate_spec(self, spec: dict) -> None:
        theme_name = spec["visual_theme"]
        theme = self.get(theme_name)
        for location in spec["locations"]:
            if location["theme"] != theme_name:
                raise ValueError(f"location {location['id']!r} uses theme {location['theme']!r}, expected {theme_name!r}")
        allowed_objects = set(theme["object_classes"])
        for obj in spec["objects"]:
            if obj["class"] not in allowed_objects:
                raise ValueError(f"object {obj['id']!r} has class {obj['class']!r}, incompatible with theme {theme_name!r}")
        allowed_npcs = set(theme["npc_archetypes"])
        for character in spec["characters"]:
            if character["archetype"] not in allowed_npcs:
                raise ValueError(f"character {character['id']!r} has archetype {character['archetype']!r}, incompatible with theme {theme_name!r}")
        allowed_roles = set(theme["hotspot_roles"])
        for entity in spec["objects"] + spec["characters"]:
            if entity["role"] not in allowed_roles:
                raise ValueError(f"entity {entity['id']!r} has role {entity['role']!r}, incompatible with theme {theme_name!r}")

    def as_document(self) -> dict:
        return {"schema_version": self.version, "themes": deepcopy(THEMES)}
