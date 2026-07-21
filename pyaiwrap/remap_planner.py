"""L5b cluster-version remap planner (Phase 1b).

Turns the Phase 0.4 per-cluster version inventory into a per-image `correspondence` for
`ClusterVersionRemap`, with the "freeze cluster-invariant colours" rule baked in so fixed-colour
objects (sky blue, foliage in an all-green cluster, ...) are protected *structurally* rather than
only caught by a human blacklist pass:

  * A chromatic colour that appears in ~all of a cluster's images is INVARIANT -> frozen: it is
    never used as a remap source and never introduced/removed by a remap. Only colours that VARY
    across the cluster's observed versions are eligible to move (green<->yellow<->brown fields).
  * A target version V' is sampled among the same cluster's observed versions with the *same*
    chromatic count, the *same* achromatic membership and an *identical frozen footprint*, so a
    remap only swaps variable colours for other variable colours actually seen in that cluster.
  * Variable sources are paired to variable targets by minimum circular hue distance, and each
    pair carries the TARGET colour's per-cluster S/V statistics (so e.g. green->brown darkens).
  * Clusters in BLACKLISTED_CLUSTERS opt out entirely (human QA verdict), for scenes the
    structural rules cannot make safe.

The planner is pure logic over injected data structures (no image I/O), so it is unit-testable.
"""

import csv
import random
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, FrozenSet, List, Optional, Tuple

from pyaiwrap.transforms import CHROMATIC_HUE_BAND, RemapTarget


def hueCenter(color: str) -> float:
    """Circular centre (degrees, 0..360) of a chromatic colour's hue band; red wraps through 0."""
    start, end = CHROMATIC_HUE_BAND[color]
    width: float = (end - start) % 360.0 or 360.0
    return (start + width / 2.0) % 360.0


def hueDistance(first: str, second: str) -> float:
    """Shortest circular distance (degrees) between two chromatic colours' band centres."""
    delta: float = abs(hueCenter(first) - hueCenter(second)) % 360.0
    return min(delta, 360.0 - delta)


@dataclass(frozen=True)
class ColorStats:
    """Saturation/value statistics of one colour as observed within one cluster (pixel-weighted)."""
    saturation_mean: float
    saturation_std: float
    value_mean: float
    value_std: float


@dataclass(frozen=True)
class ClusterVersion:
    """One colour version observed in a cluster: its chromatic set, achromatic set and support."""
    chromatic: FrozenSet[str]
    achromatic: FrozenSet[str]
    count: int


@dataclass
class ClusterProfile:
    """Everything the planner needs about one cluster: its observed versions, per-colour S/V and
    the user-assigned name (whose colour/object bindings drive name-based freezing)."""
    cluster_id: int
    versions: Tuple[ClusterVersion, ...]
    color_stats: Dict[str, ColorStats] = field(default_factory=dict)
    name: str = ""

    def totalCount(self) -> int:
        return sum(version.count for version in self.versions)


# Neutral fallback when a colour has no measured stats (keeps S/V matching a mild no-op).
DEFAULT_COLOR_STATS: ColorStats = ColorStats(saturation_mean=0.45, saturation_std=0.12,
                                             value_mean=0.5, value_std=0.15)
FREEZE_THRESHOLD: float = 0.50
MIN_SUPPORT: int = 10
# A source colour is only remapped to a target within this circular hue distance (degrees).
MAX_HUE_DISTANCE: float = 90.0

# Clusters excluded from remapping entirely, from the user's review of the QA preview grids
# (analysis/remap_qa_grids.py, 2026-07-21). These are scenes where *no* target version produced
# a plausible recolour, so the freeze rules alone cannot make them safe -- the whole cluster opts
# out of L5b rather than being protected colour by colour.
BLACKLISTED_CLUSTERS: FrozenSet[int] = frozenset(
    {9, 26, 33, 36, 46, 58, 67, 68, 69, 78, 79, 88, 91})

# --- Name-based freezing -------------------------------------------------------------------
# The user's cluster names bind colours to objects. When a colour named in the cluster's title
# describes a large fixed-colour region (a backdrop, or an object intrinsically one colour like
# sky/grass), we freeze that colour so it is never remapped -- domain knowledge the frequency
# heuristic cannot see (e.g. a blue sky that is 'only' in 82% of a cluster's images).

# Colour words that may appear in a cluster name -> our extract_colors band name.
NAME_COLOR_WORDS: Dict[str, str] = {
    "blue": "blue", "navy": "blue", "azure": "blue",
    "green": "green",
    "red": "red", "reddish": "red", "crimson": "red", "scarlet": "red",
    "brown": "brown", "brownish": "brown", "tan": "brown", "earthy": "brown", "browned": "brown",
    "yellow": "yellow", "yellowed": "yellow", "yellowish": "yellow", "gold": "yellow",
    "golden": "yellow", "gilded": "yellow",
    "orange": "orange",
    "pink": "pink",
    "violet": "purple", "purple": "purple",
    "cyan": "cyan", "turquoise": "cyan", "teal": "cyan",
    "white": "white",
}
# Modifiers to skip when scanning back from a noun for its colour (they are not colour bands).
NAME_MODIFIER_WORDS: FrozenSet[str] = frozenset({
    "dark", "light", "bright", "deep", "pale", "muted", "saturated", "lush", "warm", "cool",
    "predominantly", "mostly", "slightly", "very", "of", "the", "a", "an", "and", "or", "-",
})
# Large fixed-colour regions: an explicitly named colour bound to one of these is frozen.
NAME_BACKDROP_NOUNS: FrozenSet[str] = frozenset({
    "background", "backdrop", "sky", "skies", "lawn", "lawns", "grass", "field", "fields",
    "wall", "walls", "sea", "water", "waterside", "meadow", "ground", "vegetation", "greenery",
    "foliage", "landscape", "hills",
})
# Objects/scene cues intrinsically one colour: frozen even when the name omits the colour word.
# 'sunny' stands in for the clear blue sky it always implies in these outdoor scene titles (the
# user rarely wrote 'blue sky' explicitly); 'overcast'/'cloudy' deliberately do NOT freeze blue.
NAME_INTRINSIC_OBJECTS: Dict[str, str] = {
    "sky": "blue", "skies": "blue", "sunny": "blue",
    "grass": "green", "lawn": "green", "lawns": "green", "greenery": "green",
    "foliage": "green", "vegetation": "green", "moss": "green", "mossy": "green",
    "snow": "white",
}
# How many tokens to look back from a backdrop noun for its colour adjective.
NAME_COLOR_WINDOW: int = 4


def _tokenizeName(name: str) -> List[str]:
    cleaned: str = "".join(ch.lower() if (ch.isalnum() or ch == "-") else " " for ch in name)
    return cleaned.split()


def frozenColorsFromName(name: str) -> FrozenSet[str]:
    """Colours to freeze inferred from a cluster's title: an explicit colour word bound to a
    backdrop noun (e.g. 'green lawns', 'dark blue background'), plus intrinsic-colour objects
    (sky->blue, grass->green, snow->white) even when the colour word is implicit."""
    tokens: List[str] = _tokenizeName(name)
    frozen: set = set()
    for index, token in enumerate(tokens):
        if token in NAME_INTRINSIC_OBJECTS:
            frozen.add(NAME_INTRINSIC_OBJECTS[token])
        if token in NAME_BACKDROP_NOUNS:
            for back in range(index - 1, max(index - 1 - NAME_COLOR_WINDOW, -1), -1):
                previous: str = tokens[back]
                if previous in NAME_COLOR_WORDS:
                    frozen.add(NAME_COLOR_WORDS[previous])
                    break
                if previous not in NAME_MODIFIER_WORDS:
                    break                                    # a non-colour, non-modifier word breaks the bind
    return frozenset(frozen)


class RemapPlanner:
    """Builds a per-image `correspondence` (Dict[source_color, RemapTarget]) from a cluster
    profile, applying the freeze-invariant-colours rule. Sampling is frequency-weighted and uses
    the injected `rng` so training epochs vary while tests stay deterministic."""

    def __init__(self, freeze_threshold: float = FREEZE_THRESHOLD, min_support: int = MIN_SUPPORT,
                 use_name_freeze: bool = True, max_hue_distance: float = MAX_HUE_DISTANCE,
                 blacklist: Optional[FrozenSet[int]] = None,
                 rng: Optional[random.Random] = None) -> None:
        self.freeze_threshold: float = freeze_threshold
        self.min_support: int = min_support
        self.use_name_freeze: bool = use_name_freeze
        # Whole clusters that opt out of remapping (human QA verdict); pass frozenset() to disable.
        self.blacklist: FrozenSet[int] = (BLACKLISTED_CLUSTERS if blacklist is None else blacklist)
        # A source is only remapped to a target within this circular hue distance (degrees). It
        # stops the version structure from FORCING an unphysical cross-spectrum jump (blue sky ->
        # brown, blue sky -> green) when a target version lacks the source's colour; such a source
        # is left unchanged instead. ~90 deg still allows neighbour swaps (green<->yellow, green->
        # brown/orange for seasons/repaints).
        self.max_hue_distance: float = max_hue_distance
        self._rng: random.Random = rng if rng is not None else random.Random()

    def reseed(self, seed: int) -> None:
        """Give this planner an independent sampling stream (used to de-correlate forked
        DataLoader workers, which would otherwise all draw the same target versions)."""
        self._rng = random.Random(seed)

    def _presenceFrequency(self, profile: ClusterProfile, color: str) -> float:
        total: int = profile.totalCount()
        if total == 0:
            return 0.0
        present: int = sum(v.count for v in profile.versions if color in v.chromatic)
        return present / total

    def _frequencyFrozen(self, profile: ClusterProfile) -> FrozenSet[str]:
        chromatic_colors: set = set()
        for version in profile.versions:
            chromatic_colors |= set(version.chromatic)
        return frozenset(color for color in chromatic_colors
                         if self._presenceFrequency(profile, color) >= self.freeze_threshold)

    def frozenColors(self, profile: ClusterProfile) -> FrozenSet[str]:
        """Colours never remapped in this cluster: the union of colours present in ~all of the
        cluster's images (>= freeze_threshold) and colours the cluster NAME binds to a fixed-colour
        object/backdrop (sky->blue, 'green lawns', ...). Name freezing catches fixed colours that
        fall below the frequency threshold (e.g. a sky present in only 82% of a cluster)."""
        frozen: FrozenSet[str] = self._frequencyFrozen(profile)
        if self.use_name_freeze and profile.name:
            frozen = frozen | frozenColorsFromName(profile.name)
        return frozen

    def _eligibleTargets(self, profile: ClusterProfile, source: ClusterVersion,
                         frozen: FrozenSet[str]) -> List[ClusterVersion]:
        frozen_here: FrozenSet[str] = frozen & source.chromatic
        source_size: int = len(source.chromatic)
        eligible: List[ClusterVersion] = []
        for candidate in profile.versions:
            if candidate.chromatic == source.chromatic:
                continue                                     # the source version itself
            if candidate.achromatic != source.achromatic:
                continue                                     # keep achromatic structure fixed
            if len(candidate.chromatic) != source_size:
                continue                                     # keep chromatic count fixed
            if (frozen & candidate.chromatic) != frozen_here:
                continue                                     # frozen footprint must match exactly
            if candidate.count < self.min_support:
                continue
            eligible.append(candidate)
        return eligible

    def _statsFor(self, profile: ClusterProfile, color: str) -> ColorStats:
        return profile.color_stats.get(color, DEFAULT_COLOR_STATS)

    @staticmethod
    def _pairByHue(sources: List[str], targets: List[str]) -> Dict[str, str]:
        """Bijection sources->targets minimising total circular hue distance (small sets)."""
        best_pairing: Dict[str, str] = {}
        best_cost: float = float("inf")
        for order in permutations(targets):
            cost: float = sum(hueDistance(s, t) for s, t in zip(sources, order))
            if cost < best_cost:
                best_cost = cost
                best_pairing = {s: t for s, t in zip(sources, order)}
        return best_pairing

    def planCorrespondence(self, profile: ClusterProfile,
                           source: ClusterVersion) -> Dict[str, RemapTarget]:
        """Sample one eligible target version and build the source->RemapTarget correspondence.
        Returns {} when nothing can be safely remapped (blacklisted cluster, all chromatics
        frozen, or no eligible target version)."""
        if profile.cluster_id in self.blacklist:
            return {}
        frozen: FrozenSet[str] = self.frozenColors(profile)
        frozen_here: FrozenSet[str] = frozen & source.chromatic
        variable_sources: List[str] = sorted(source.chromatic - frozen_here)
        if not variable_sources:
            return {}                                        # nothing varies -> nothing to remap

        candidates: List[ClusterVersion] = self._eligibleTargets(profile, source, frozen)
        if not candidates:
            return {}

        weights: List[int] = [candidate.count for candidate in candidates]
        chosen: ClusterVersion = self._rng.choices(candidates, weights=weights, k=1)[0]
        variable_targets: List[str] = sorted(chosen.chromatic - frozen_here)

        pairing: Dict[str, str] = self._pairByHue(variable_sources, variable_targets)
        correspondence: Dict[str, RemapTarget] = {}
        for source_color, target_color in pairing.items():
            if target_color == source_color:
                continue
            if hueDistance(source_color, target_color) > self.max_hue_distance:
                continue                                     # skip forced cross-spectrum jumps
            stats: ColorStats = self._statsFor(profile, target_color)
            correspondence[source_color] = RemapTarget(
                target_color=target_color,
                saturation_mean=stats.saturation_mean, saturation_std=stats.saturation_std,
                value_mean=stats.value_mean, value_std=stats.value_std)
        return correspondence


def _splitColorSet(pipe_joined: str) -> FrozenSet[str]:
    return frozenset(part for part in pipe_joined.split("|") if part)


def loadClusterProfiles(version_inventory_path: str, color_sv_path: str,
                        cluster_names_path: Optional[str] = None,
                        count_column: str = "n_train") -> Dict[int, ClusterProfile]:
    """Build ClusterProfile objects from the Phase 0.4 version inventory + the per-cluster/colour
    S/V table (extract_cluster_color_sv.py). `count_column` selects the split whose per-version
    support weights the planner (default n_train, the train-time augmentation distribution).
    `cluster_names_path` (cluster_name_proposals_k98.csv) supplies each cluster's `final_name`,
    enabling name-based freezing of fixed-colour objects."""
    names_by_cluster: Dict[int, str] = {}
    if cluster_names_path is not None:
        with open(cluster_names_path, newline="") as handle:
            for row in csv.DictReader(handle):
                names_by_cluster[int(row["cluster"])] = row.get("final_name", "") or ""

    color_stats: Dict[int, Dict[str, ColorStats]] = {}
    with open(color_sv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            cluster_id: int = int(row["cluster_k98"])
            color_stats.setdefault(cluster_id, {})[row["color"]] = ColorStats(
                saturation_mean=float(row["saturation_mean"]),
                saturation_std=float(row["saturation_std"]),
                value_mean=float(row["value_mean"]),
                value_std=float(row["value_std"]))

    versions_by_cluster: Dict[int, List[ClusterVersion]] = {}
    with open(version_inventory_path, newline="") as handle:
        for row in csv.DictReader(handle):
            count: int = int(row[count_column])
            if count <= 0:
                continue                                     # no support in the selected split
            cluster_id = int(row["cluster_k98"])
            achromatic: FrozenSet[str] = _splitColorSet(row["achromatic_set"])
            chromatic: FrozenSet[str] = _splitColorSet(row["version"]) - achromatic
            versions_by_cluster.setdefault(cluster_id, []).append(
                ClusterVersion(chromatic=chromatic, achromatic=achromatic, count=count))

    profiles: Dict[int, ClusterProfile] = {}
    for cluster_id, versions in versions_by_cluster.items():
        profiles[cluster_id] = ClusterProfile(
            cluster_id=cluster_id, versions=tuple(versions),
            color_stats=color_stats.get(cluster_id, {}),
            name=names_by_cluster.get(cluster_id, ""))
    return profiles
