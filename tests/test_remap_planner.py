"""Tests for the L5b cluster-version remap planner (freeze-invariant colours rule).

Pure-logic tests: the planner consumes an in-memory cluster profile (version inventory +
per-color S/V stats) and emits a correspondence for ClusterVersionRemap. No image I/O here.
"""

import random

import pytest

from pyaiwrap.remap_planner import (BLACKLISTED_CLUSTERS, ColorStats, ClusterProfile,
                                     ClusterVersion, RemapPlanner, frozenColorsFromName, hueCenter)
from pyaiwrap.transforms import RemapTarget


def makeStats(saturation: float = 0.4, value: float = 0.5) -> ColorStats:
    return ColorStats(saturation_mean=saturation, saturation_std=0.1,
                      value_mean=value, value_std=0.1)


def fieldCluster() -> ClusterProfile:
    """A 'field' cluster: sky-blue is invariant (in every version), the ground colour
    varies green/yellow/brown. Achromatic membership {gray, white} is constant."""
    achromatic: frozenset = frozenset({"gray", "white"})
    versions: tuple = (
        ClusterVersion(chromatic=frozenset({"blue", "green"}), achromatic=achromatic, count=120),
        ClusterVersion(chromatic=frozenset({"blue", "yellow"}), achromatic=achromatic, count=80),
        ClusterVersion(chromatic=frozenset({"blue", "brown"}), achromatic=achromatic, count=40),
    )
    color_stats: dict = {
        "blue": makeStats(0.5, 0.6), "green": makeStats(0.45, 0.5),
        "yellow": makeStats(0.55, 0.7), "brown": makeStats(0.4, 0.4),
    }
    return ClusterProfile(cluster_id=1, versions=versions, color_stats=color_stats)


def testHueCenterHandlesRedWrap() -> None:
    assert hueCenter("orange") == pytest.approx(30.0)
    # red spans 345..15, its circular centre is 0/360
    assert hueCenter("red") % 360.0 == pytest.approx(0.0)


def testFrozenColorsDetectsInvariantColour() -> None:
    planner: RemapPlanner = RemapPlanner(freeze_threshold=0.90)
    frozen: frozenset = planner.frozenColors(fieldCluster())
    assert "blue" in frozen                       # present in 100% of versions
    assert "green" not in frozen and "yellow" not in frozen and "brown" not in frozen


def testFrozenColourIsNeverASource() -> None:
    planner: RemapPlanner = RemapPlanner(freeze_threshold=0.90, min_support=10,
                                         rng=random.Random(0))
    profile: ClusterProfile = fieldCluster()
    source: ClusterVersion = profile.versions[0]   # {blue, green}
    for _ in range(20):
        correspondence: dict = planner.planCorrespondence(profile, source)
        assert "blue" not in correspondence         # sky is structurally protected
        assert set(correspondence) <= {"green"}     # only the variable ground colour moves


def testCorrespondenceRemapsVariableColourToAnotherObservedVersion() -> None:
    planner: RemapPlanner = RemapPlanner(freeze_threshold=0.90, min_support=10,
                                         rng=random.Random(0))
    profile: ClusterProfile = fieldCluster()
    source: ClusterVersion = profile.versions[0]   # {blue, green}
    seen_targets: set = set()
    for _ in range(60):
        correspondence: dict = planner.planCorrespondence(profile, source)
        if not correspondence:
            continue
        target: RemapTarget = correspondence["green"]
        assert target.target_color in {"yellow", "brown"}   # the other observed ground colours
        # RemapTarget carries the TARGET colour's cluster S/V stats
        expected: ColorStats = profile.color_stats[target.target_color]
        assert target.saturation_mean == pytest.approx(expected.saturation_mean)
        assert target.value_mean == pytest.approx(expected.value_mean)
        seen_targets.add(target.target_color)
    assert seen_targets == {"yellow", "brown"}      # frequency-weighted sampling reaches both


def testMinSupportFiltersRareTargets() -> None:
    planner: RemapPlanner = RemapPlanner(freeze_threshold=0.90, min_support=100,
                                         rng=random.Random(0))
    profile: ClusterProfile = fieldCluster()
    source: ClusterVersion = profile.versions[0]   # {blue, green}; only {blue,yellow}=80,{blue,brown}=40 remain
    for _ in range(30):
        correspondence: dict = planner.planCorrespondence(profile, source)
        assert correspondence == {}                 # no target clears support>=100


def testFrozenSetMismatchIsNotEligible() -> None:
    """A candidate that would turn a variable colour into a frozen colour (or drop the frozen
    colour) must be rejected — otherwise a field could be recoloured sky-blue."""
    planner: RemapPlanner = RemapPlanner(freeze_threshold=0.90, min_support=1,
                                         rng=random.Random(0))
    achromatic: frozenset = frozenset({"gray"})
    versions: tuple = (
        ClusterVersion(chromatic=frozenset({"blue", "green"}), achromatic=achromatic, count=100),
        ClusterVersion(chromatic=frozenset({"blue", "yellow"}), achromatic=achromatic, count=100),
        # blue absent here: green+yellow. blue is frozen (in 2/3 weighted heavily), so this
        # version has a different frozen footprint and must not be a target for {blue,green}.
        ClusterVersion(chromatic=frozenset({"green", "yellow"}), achromatic=achromatic, count=5),
    )
    color_stats: dict = {c: makeStats() for c in ("blue", "green", "yellow")}
    profile: ClusterProfile = ClusterProfile(cluster_id=2, versions=versions, color_stats=color_stats)
    source: ClusterVersion = versions[0]            # {blue, green}
    for _ in range(30):
        correspondence: dict = planner.planCorrespondence(profile, source)
        if correspondence:
            assert correspondence["green"].target_color == "yellow"   # never 'blue'


def testPairingUsesMinimumHueDistance() -> None:
    """Two variable sources pair to the two variable targets by nearest hue."""
    planner: RemapPlanner = RemapPlanner(freeze_threshold=0.999, min_support=1,
                                         rng=random.Random(0))
    achromatic: frozenset = frozenset({"black"})
    versions: tuple = (
        ClusterVersion(chromatic=frozenset({"red", "green"}), achromatic=achromatic, count=50),
        ClusterVersion(chromatic=frozenset({"orange", "cyan"}), achromatic=achromatic, count=50),
    )
    color_stats: dict = {c: makeStats() for c in ("red", "green", "orange", "cyan")}
    profile: ClusterProfile = ClusterProfile(cluster_id=3, versions=versions, color_stats=color_stats)
    source: ClusterVersion = versions[0]            # {red, green}
    correspondence: dict = planner.planCorrespondence(profile, source)
    # red(~0) is nearest orange(~30); green(~117) is nearest cyan(~182)
    assert correspondence["red"].target_color == "orange"
    assert correspondence["green"].target_color == "cyan"


def testNameFreezeExplicitColorBoundToBackdrop() -> None:
    assert "green" in frozenColorsFromName("green cemetery lawns")
    assert "blue" in frozenColorsFromName("postcards under saturated blue skies")
    assert "brown" in frozenColorsFromName("a suit against a brown background")


def testNameFreezeIntrinsicObjectWithoutColorWord() -> None:
    # sky is intrinsically blue, grass/greenery intrinsically green -- frozen without a colour word
    assert frozenColorsFromName("a warehouse under the open sky") == frozenset({"blue"})
    assert "green" in frozenColorsFromName("grass and trees in the wild")


def testNameFreezeSunnyImpliesBlueSky() -> None:
    # 'sunny day/weather' implies a clear blue sky even though the title never writes 'blue'
    assert "blue" in frozenColorsFromName("a warehouse on a sunny day")
    assert "blue" in frozenColorsFromName("folk festivals in sunny weather")
    # overcast/cloudy must NOT freeze blue (grey/white sky)
    assert "blue" not in frozenColorsFromName("a rural landscape on a slightly overcast day")


def testNameFreezeDoesNotFreezeSubjectColour() -> None:
    # 'red' modifies houses (a subject, not a backdrop) -> not frozen; the green lawn IS frozen
    frozen: frozenset = frozenColorsFromName("red country houses with white trim on green lawns")
    assert "red" not in frozen
    assert "green" in frozen


def testNameFreezeIgnoresHarvestFieldWithoutColour() -> None:
    # 'field' with no bound colour word must not be force-frozen green (harvest fields are golden),
    # so the golden field stays remappable; only the sunny-day sky (blue) is frozen.
    frozen: frozenset = frozenColorsFromName("a field ready for harvest on a sunny day")
    assert "green" not in frozen
    assert frozen == frozenset({"blue"})


def testFrozenColorsUnionsNameAndFrequency() -> None:
    planner: RemapPlanner = RemapPlanner(freeze_threshold=0.90)
    profile: ClusterProfile = fieldCluster()               # blue freq=1.0 -> frozen by frequency
    profile.name = "a meadow with green grass and a red barn"
    frozen: frozenset = planner.frozenColors(profile)
    assert "blue" in frozen                                 # from frequency
    assert "green" in frozen                                # from the name (green grass)


def testNameFreezeDisabled() -> None:
    # threshold > 1.0 => frequency freezes nothing, so only name freezing could add colours
    planner: RemapPlanner = RemapPlanner(freeze_threshold=1.01, use_name_freeze=False)
    profile: ClusterProfile = fieldCluster()
    profile.name = "green fields under blue skies"
    assert planner.frozenColors(profile) == frozenset()     # nothing frozen when disabled
    enabled: RemapPlanner = RemapPlanner(freeze_threshold=1.01, use_name_freeze=True)
    assert enabled.frozenColors(profile) == frozenset({"green", "blue"})


def testHueCapSkipsForcedCrossSpectrumJump() -> None:
    """A blue-sky source with only a brown target available must be left unchanged, not forced
    into blue->brown (a browned sky). blue->brown ~= 165 deg > the 90 deg cap."""
    planner: RemapPlanner = RemapPlanner(freeze_threshold=1.01, use_name_freeze=False,
                                         min_support=1, max_hue_distance=90.0, rng=random.Random(0))
    achromatic: frozenset = frozenset({"gray"})
    versions: tuple = (
        ClusterVersion(chromatic=frozenset({"blue", "red"}), achromatic=achromatic, count=50),
        ClusterVersion(chromatic=frozenset({"brown", "orange"}), achromatic=achromatic, count=50),
    )
    color_stats: dict = {c: makeStats() for c in ("blue", "red", "brown", "orange")}
    profile: ClusterProfile = ClusterProfile(cluster_id=9, versions=versions, color_stats=color_stats)
    source: ClusterVersion = versions[0]                    # {blue, red}
    for _ in range(20):
        correspondence: dict = planner.planCorrespondence(profile, source)
        assert "blue" not in correspondence                 # blue->brown/orange both > 90 deg
        if "red" in correspondence:                         # red->brown(~32) or red->orange(~30) ok
            assert correspondence["red"].target_color in {"brown", "orange"}


def testHueCapAllowsNeighbourSwaps() -> None:
    planner: RemapPlanner = RemapPlanner(freeze_threshold=1.01, use_name_freeze=False,
                                         min_support=1, max_hue_distance=90.0, rng=random.Random(0))
    achromatic: frozenset = frozenset({"gray"})
    versions: tuple = (
        ClusterVersion(chromatic=frozenset({"green"}), achromatic=achromatic, count=50),
        ClusterVersion(chromatic=frozenset({"yellow"}), achromatic=achromatic, count=50),
    )
    profile: ClusterProfile = ClusterProfile(
        cluster_id=10, versions=versions,
        color_stats={"green": makeStats(), "yellow": makeStats()})
    correspondence: dict = planner.planCorrespondence(profile, versions[0])   # green->yellow ~60 deg
    assert correspondence["green"].target_color == "yellow"


def testAllChromaticsFrozenYieldsNoRemap() -> None:
    planner: RemapPlanner = RemapPlanner(freeze_threshold=0.5, min_support=1,
                                         rng=random.Random(0))
    achromatic: frozenset = frozenset({"gray"})
    versions: tuple = (
        ClusterVersion(chromatic=frozenset({"blue"}), achromatic=achromatic, count=100),
        ClusterVersion(chromatic=frozenset({"blue"}), achromatic=frozenset({"white"}), count=100),
    )
    profile: ClusterProfile = ClusterProfile(cluster_id=4, versions=versions,
                                             color_stats={"blue": makeStats()})
    source: ClusterVersion = versions[0]
    assert planner.planCorrespondence(profile, source) == {}


def testBlacklistedClusterIsNeverRemapped() -> None:
    """A cluster rejected in the human QA pass yields no remap, however remappable it looks."""
    planner: RemapPlanner = RemapPlanner(freeze_threshold=0.90, min_support=10,
                                         rng=random.Random(0))
    profile: ClusterProfile = fieldCluster()
    remappable: dict = planner.planCorrespondence(profile, profile.versions[0])
    assert remappable                                       # cluster 1 is not blacklisted

    blacklisted: ClusterProfile = ClusterProfile(
        cluster_id=sorted(BLACKLISTED_CLUSTERS)[0], versions=profile.versions,
        color_stats=profile.color_stats)
    assert planner.planCorrespondence(blacklisted, profile.versions[0]) == {}


def testBlacklistIsInjectable() -> None:
    profile: ClusterProfile = fieldCluster()
    blocked: RemapPlanner = RemapPlanner(freeze_threshold=0.90, min_support=10,
                                         blacklist=frozenset({profile.cluster_id}),
                                         rng=random.Random(0))
    allowed: RemapPlanner = RemapPlanner(freeze_threshold=0.90, min_support=10,
                                         blacklist=frozenset(), rng=random.Random(0))
    assert blocked.planCorrespondence(profile, profile.versions[0]) == {}
    assert allowed.planCorrespondence(profile, profile.versions[0])
