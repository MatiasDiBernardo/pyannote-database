"""Annotation versioning tests

`versions:` / `version:` are declared at the TASK level (shared by every
protocol of the task); a protocol is versioned only if it uses the
`{version}` placeholder, and a protocol may override the task-level default.

Primary convention (module fixture): `versions:` as a list of names, with
`{version}` substituted into per-version folder paths, and `original` as the
default version.

Also covered (standalone fixtures):
- task-level inheritance, per-protocol override, and unversioned protocols
  under a versioned task;
- the mapping form with a sparse per-(subset, key) overlay (advanced case);
- default resolution (explicit `version:` > `original` > first listed);
- flat (unversioned) backward compatibility;
- load-time errors and the no-silent-fallback guarantee.
"""

import pytest

from pyannote.database import registry as global_registry
from pyannote.database.registry import Registry


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _duration(file):
    return file["annotation"].get_timeline().duration()


# --------------------------------------------------------------------------- #
# Primary convention: task-level list form + {version} folders + `original`
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def versioned_dataset(tmp_path_factory):
    """Two versions declared once at the task level, in per-version folders.

    - original (default): uris u1 + u2, 1.0 s segments
    - refined: trains on u1 only (its own list folder), 2.0 s segments
    - Flat: a protocol with no {version} -> stays unversioned under the task
    """
    root = tmp_path_factory.mktemp("versioned")

    _write(root / "lists" / "original" / "train.lst", "u1\nu2\n")
    _write(root / "lists" / "refined" / "train.lst", "u1\n")

    for uri in ("u1", "u2"):
        _write(
            root / "rttm" / "original" / f"{uri}.rttm",
            f"SPEAKER {uri} 1 0.0 1.0 <NA> <NA> spk_{uri} <NA> <NA>\n",
        )
        _write(root / "uem" / f"{uri}.uem", f"{uri} 1 0.000 10.000\n")
    _write(
        root / "rttm" / "refined" / "u1.rttm",
        "SPEAKER u1 1 0.0 2.0 <NA> <NA> spk_u1 <NA> <NA>\n",
    )

    _write(
        root / "database.yml",
        """
Protocols:
  VersionedTestDB:
    SpeakerDiarization:
      versions: [original, refined]     # task-level; default is "original"
      Raw:
        scope: file
        train:
          uri: lists/{version}/train.lst
          annotation: rttm/{version}/{uri}.rttm
          annotated: uem/{uri}.uem
      Flat:                             # no {version} -> stays unversioned
        scope: file
        train:
          uri: lists/original/train.lst
          annotation: rttm/original/{uri}.rttm
          annotated: uem/{uri}.uem
  X:
    SpeakerDiarization:
      VersionedTestCompound:
        train:
          VersionedTestDB.SpeakerDiarization.Raw@refined: [train]
""",
    )

    # the compound (X) member resolution goes through the global registry
    global_registry.load_database(root / "database.yml")
    return root


def test_switching_version_changes_annotation(versioned_dataset):
    """Same uri, different version -> different annotation content."""
    original = global_registry.get_protocol("VersionedTestDB.SpeakerDiarization.Raw@original")
    refined = global_registry.get_protocol("VersionedTestDB.SpeakerDiarization.Raw@refined")

    original_files = {f["uri"]: f for f in original.train()}
    refined_files = {f["uri"]: f for f in refined.train()}

    assert _duration(original_files["u1"]) == 1.0
    assert _duration(refined_files["u1"]) == 2.0

    assert original_files["u1"]["annotation_version"] == "original"
    assert refined_files["u1"]["annotation_version"] == "refined"


def test_version_changes_which_data(versioned_dataset):
    """Each version reads its own {version} list folder -> different uris."""
    original = global_registry.get_protocol("VersionedTestDB.SpeakerDiarization.Raw@original")
    refined = global_registry.get_protocol("VersionedTestDB.SpeakerDiarization.Raw@refined")

    assert sorted(f["uri"] for f in original.train()) == ["u1", "u2"]
    assert sorted(f["uri"] for f in refined.train()) == ["u1"]


def test_omitted_version_defaults_to_original(versioned_dataset):
    """Bare protocol name resolves to the conventional 'original' default."""
    default = global_registry.get_protocol("VersionedTestDB.SpeakerDiarization.Raw")
    files = list(default.train())

    assert sorted(f["uri"] for f in files) == ["u1", "u2"]
    assert all(f["annotation_version"] == "original" for f in files)
    assert default.name == "VersionedTestDB.SpeakerDiarization.Raw"


def test_unversioned_protocol_under_versioned_task_stays_flat(versioned_dataset):
    """A protocol with no {version} is not expanded, even under a versioned task."""
    database = global_registry.get_database("VersionedTestDB")
    protocols = database.get_protocols("SpeakerDiarization")
    assert "Flat" in protocols
    assert "Flat@original" not in protocols
    assert "Flat@refined" not in protocols

    flat = global_registry.get_protocol("VersionedTestDB.SpeakerDiarization.Flat")
    for file in flat.train():
        # no annotation_version key is introduced (iterate keys: does not
        # trigger lazy evaluation)
        assert "annotation_version" not in set(file)
        assert _duration(file) == 1.0


def test_versions_are_listed_as_protocols(versioned_dataset):
    protocols = global_registry.get_database("VersionedTestDB").get_protocols(
        "SpeakerDiarization"
    )
    assert "Raw" in protocols
    assert "Raw@original" in protocols
    assert "Raw@refined" in protocols


def test_unknown_version_is_clear_error(versioned_dataset):
    with pytest.raises(ValueError, match="version 'nope'.*available versions.*original"):
        global_registry.get_protocol("VersionedTestDB.SpeakerDiarization.Raw@nope")


def test_compound_member_with_version(versioned_dataset):
    """X meta-protocol members can pin an annotation version with @."""
    compound = global_registry.get_protocol(
        "X.SpeakerDiarization.VersionedTestCompound"
    )
    files = list(compound.train())

    assert sorted(f["uri"] for f in files) == ["u1"]
    assert files[0]["annotation_version"] == "refined"
    assert _duration(files[0]) == 2.0


# --------------------------------------------------------------------------- #
# Task-level inheritance vs per-protocol override
# --------------------------------------------------------------------------- #


def test_protocol_level_versions_override_task(tmp_path):
    """A protocol may declare its own `versions`, overriding the task default."""
    for v in ("original", "refined", "special"):
        _write(tmp_path / "lists" / v / "train.lst", "u1\n")
        _write(
            tmp_path / "rttm" / v / "u1.rttm",
            f"SPEAKER u1 1 0.0 1.0 <NA> <NA> spk_{v} <NA> <NA>\n",
        )
    _write(
        tmp_path / "database.yml",
        """
Protocols:
  OverrideDB:
    SpeakerDiarization:
      versions: [original, refined]     # task default
      Raw:
        scope: file
        train:
          uri: lists/{version}/train.lst
          annotation: rttm/{version}/{uri}.rttm
      Special:
        scope: file
        versions: [special]             # overrides the task default
        train:
          uri: lists/{version}/train.lst
          annotation: rttm/{version}/{uri}.rttm
""",
    )
    registry = Registry()
    registry.load_database(tmp_path / "database.yml")

    protocols = registry.get_database("OverrideDB").get_protocols("SpeakerDiarization")
    assert "Raw@original" in protocols and "Raw@refined" in protocols
    assert "Special@special" in protocols
    assert "Special@original" not in protocols   # task default did not leak in


# --------------------------------------------------------------------------- #
# Default-version resolution
# --------------------------------------------------------------------------- #


def test_default_is_original_even_when_not_first(tmp_path):
    """With no explicit `version:`, 'original' is the default even if listed later."""
    for v in ("only_words", "original"):
        _write(tmp_path / "lists" / v / "train.lst", "u1\n")
        _write(
            tmp_path / "rttm" / v / "u1.rttm",
            f"SPEAKER u1 1 0.0 1.0 <NA> <NA> spk_{v} <NA> <NA>\n",
        )
    _write(
        tmp_path / "database.yml",
        """
Protocols:
  OriginalDefaultDB:
    SpeakerDiarization:
      versions: [only_words, original]   # 'original' is NOT first
      Raw:
        scope: file
        train:
          uri: lists/{version}/train.lst
          annotation: rttm/{version}/{uri}.rttm
""",
    )
    registry = Registry()
    registry.load_database(tmp_path / "database.yml")

    default = list(registry.get_protocol("OriginalDefaultDB.SpeakerDiarization.Raw").train())
    assert default[0]["annotation_version"] == "original"


def test_default_is_first_when_no_original(tmp_path):
    """With no `original` and no explicit `version:`, default is the first listed."""
    for v in ("alpha", "beta"):
        _write(tmp_path / "lists" / v / "train.lst", "u1\n")
        _write(
            tmp_path / "rttm" / v / "u1.rttm",
            f"SPEAKER u1 1 0.0 1.0 <NA> <NA> spk_{v} <NA> <NA>\n",
        )
    _write(
        tmp_path / "database.yml",
        """
Protocols:
  FirstDefaultDB:
    SpeakerDiarization:
      versions: [alpha, beta]
      Raw:
        scope: file
        train:
          uri: lists/{version}/train.lst
          annotation: rttm/{version}/{uri}.rttm
""",
    )
    registry = Registry()
    registry.load_database(tmp_path / "database.yml")

    default = list(registry.get_protocol("FirstDefaultDB.SpeakerDiarization.Raw").train())
    assert default[0]["annotation_version"] == "alpha"


def test_explicit_version_key_overrides_original(tmp_path):
    """An explicit task-level `version:` wins over the 'original' convention."""
    for v in ("original", "refined"):
        _write(tmp_path / "lists" / v / "train.lst", "u1\n")
        _write(
            tmp_path / "rttm" / v / "u1.rttm",
            f"SPEAKER u1 1 0.0 1.0 <NA> <NA> spk_{v} <NA> <NA>\n",
        )
    _write(
        tmp_path / "database.yml",
        """
Protocols:
  ExplicitDefaultDB:
    SpeakerDiarization:
      version: refined                 # explicit -> wins over 'original'
      versions: [original, refined]
      Raw:
        scope: file
        train:
          uri: lists/{version}/train.lst
          annotation: rttm/{version}/{uri}.rttm
""",
    )
    registry = Registry()
    registry.load_database(tmp_path / "database.yml")

    default = list(registry.get_protocol("ExplicitDefaultDB.SpeakerDiarization.Raw").train())
    assert default[0]["annotation_version"] == "refined"


# --------------------------------------------------------------------------- #
# Mapping form with a per-subset overlay (advanced case)
# --------------------------------------------------------------------------- #


def test_mapping_form_overlay_overrides_data(tmp_path):
    """Mapping form: a version reuses an existing fixed filename via an overlay,
    instead of living in a {version}/ folder."""
    _write(tmp_path / "lists" / "train.lst", "u1\nu2\n")
    _write(tmp_path / "lists" / "train.sdm.lst", "u1\n")
    for v in ("original", "only_words"):
        for uri in ("u1", "u2"):
            _write(
                tmp_path / "rttm" / v / f"{uri}.rttm",
                f"SPEAKER {uri} 1 0.0 1.0 <NA> <NA> spk_{v} <NA> <NA>\n",
            )
    _write(
        tmp_path / "database.yml",
        """
Protocols:
  OverlayDB:
    SpeakerDiarization:
      versions:
        original: {}
        only_words:
          train:
            uri: lists/train.sdm.lst
      Raw:
        scope: file
        train:
          uri: lists/train.lst
          annotation: rttm/{version}/{uri}.rttm
""",
    )
    registry = Registry()
    registry.load_database(tmp_path / "database.yml")

    original = registry.get_protocol("OverlayDB.SpeakerDiarization.Raw@original")
    only_words = registry.get_protocol("OverlayDB.SpeakerDiarization.Raw@only_words")
    assert sorted(f["uri"] for f in original.train()) == ["u1", "u2"]
    assert sorted(f["uri"] for f in only_words.train()) == ["u1"]
    # default resolves to 'original' (present) with no explicit `version:`
    default = registry.get_protocol("OverlayDB.SpeakerDiarization.Raw")
    assert next(iter(default.train()))["annotation_version"] == "original"


# --------------------------------------------------------------------------- #
# Load-time errors and the no-silent-fallback guarantee
# --------------------------------------------------------------------------- #


def test_version_placeholder_without_versions_block_fails(tmp_path):
    _write(tmp_path / "lists" / "train.lst", "u1\n")
    _write(
        tmp_path / "database.yml",
        """
Protocols:
  BrokenVersionedDB:
    SpeakerDiarization:
      Raw:
        scope: file
        train:
          uri: lists/train.lst
          annotation: rttm/{version}/{uri}.rttm
""",
    )
    registry = Registry()
    with pytest.raises(ValueError, match="{version}.*versions"):
        registry.load_database(tmp_path / "database.yml")


def test_default_version_must_be_declared(tmp_path):
    _write(tmp_path / "lists" / "train.lst", "u1\n")
    _write(
        tmp_path / "database.yml",
        """
Protocols:
  BrokenDefaultDB:
    SpeakerDiarization:
      version: v3
      versions: [v1, v2]
      Raw:
        scope: file
        train:
          uri: lists/train.lst
          annotation: rttm/{version}/{uri}.rttm
""",
    )
    registry = Registry()
    with pytest.raises(ValueError, match="default version 'v3' is not declared"):
        registry.load_database(tmp_path / "database.yml")


def test_missing_version_file_fails_loudly(tmp_path):
    """No silent fallback: a version pointing at a missing file fails."""
    _write(tmp_path / "lists" / "refined" / "train.lst", "u1\nu2\n")  # includes u2 ...
    _write(
        tmp_path / "rttm" / "refined" / "u1.rttm",  # ... but only u1 has an rttm
        "SPEAKER u1 1 0.0 1.0 <NA> <NA> spk1 <NA> <NA>\n",
    )
    for uri in ("u1", "u2"):
        _write(tmp_path / "uem" / f"{uri}.uem", f"{uri} 1 0.000 10.000\n")
    _write(
        tmp_path / "database.yml",
        """
Protocols:
  MissingFileDB:
    SpeakerDiarization:
      versions: [refined]
      Raw:
        scope: file
        train:
          uri: lists/{version}/train.lst
          annotation: rttm/{version}/{uri}.rttm
          annotated: uem/{uri}.uem
""",
    )
    registry = Registry()
    registry.load_database(tmp_path / "database.yml")
    protocol = registry.get_protocol("MissingFileDB.SpeakerDiarization.Raw@refined")

    with pytest.raises(FileNotFoundError):
        for file in protocol.train():
            file["annotation"]  # u2 -> rttm/refined/u2.rttm does not exist
