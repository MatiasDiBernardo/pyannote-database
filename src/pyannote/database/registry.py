#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2022- CNRS
# Copyright (c) 2022- Université Paul Sabatier

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr
# Alexis PLAQUET

from enum import Enum
import os
from pathlib import Path
from typing import Dict, List, Set, Optional, Text, Tuple, Type, Union
import warnings

from pyannote.database.protocol.protocol import Preprocessors, Protocol
from .custom import create_protocol, get_init
from .database import Database
import yaml


# controls what to do in case of protocol name conflict
class LoadingMode(Enum):
    OVERRIDE = 0  # override existing protocol
    KEEP = 1  # keep existing protocol
    ERROR = 2  # raise an error


# --------------------------------------------------------------------------- #
# Annotation versioning
#
# The set of annotation versions is declared once at the TASK level (shared by
# every protocol of the task). In the common case each version differs only by
# files organized in per-version folders, so `versions:` is just a list of
# names and `{version}` is substituted into the paths:
#
#   SpeakerDiarization:
#     versions: [original, only_words]           # shared; default is "original"
#     Raw:
#       scope: database
#       train:
#         uri: annotations/lists/{version}/train.lst
#         annotation: annotations/rttm/{version}/{uri}.rttm
#         annotated: annotations/uem/{uri}.uem    # shared: no {version}
#     Benchmark:
#       scope: database
#       test: { ... }                             # same versions, no repetition
#
# A protocol is versioned only if it uses the {version} placeholder; a protocol
# with no {version} stays unversioned even under a versioned task. `versions` /
# `version` may also be set on an individual protocol to override the task-level
# default.
#
# When a version needs to override specific entries (rather than live in a
# {version} folder), use the mapping form, whose values are sparse
# per-(subset, key) overlays ({} means "no override"):
#
#   versions:
#     original: {}
#     only_words:
#       train: {uri: annotations/lists/train.sdm.lst}
#
# The default version (used when no "@version" is requested) is the explicit
# `version:` key if given, else "original" if present, else the first listed.
# `{version}` is substituted eagerly at configuration-load time; one protocol
# is registered per version as "Raw@<version>", plus "Raw" for the default.
# There is NO fallback: an unknown "@version" errors, and a version whose
# substituted path does not exist fails like any wrong path.
# --------------------------------------------------------------------------- #

_VERSION_PLACEHOLDER = "{version}"

# conventional default version name, used when a protocol declares `versions:`
# but no explicit `version:` default
_DEFAULT_VERSION = "original"


def _substitute_version(value, version: Text):
    """Replace the {version} placeholder in a string (pass anything else through)."""
    if isinstance(value, str):
        return value.replace(_VERSION_PLACEHOLDER, version)
    return value


def _uses_version_placeholder(protocol_entries: Dict) -> bool:
    """Check whether any subset entry uses the {version} placeholder."""
    for value in protocol_entries.values():
        if not isinstance(value, dict):
            continue
        for key, entry in value.items():
            if _VERSION_PLACEHOLDER in str(key) or (
                isinstance(entry, str) and _VERSION_PLACEHOLDER in entry
            ):
                return True
    return False


def _apply_version(entries: Dict, overlay, version: Text, where: Text) -> Dict:
    """Resolve one annotation version of a protocol block.

    Substitutes {version} in every (subset, key) template, then applies the
    version's sparse per-(subset, key) overlay on top.

    Parameters
    ----------
    entries : dict
        Protocol block without its `version` / `versions` keys.
    overlay : dict or None
        This version's entry in `versions:` (sparse {subset: {key: value}}).
    version : str
        Version name.
    where : str
        "Database.Task.Protocol (file.yml)" context, used in error messages.

    Returns
    -------
    resolved : dict
        Protocol block for this specific version.
    """

    if overlay is None:
        overlay = dict()
    if not isinstance(overlay, dict):
        msg = (
            f"Version '{version}' of {where} must be a mapping "
            f"(subset -> entry overrides) or empty, got: {overlay!r}"
        )
        raise ValueError(msg)

    resolved = dict()
    for key, value in entries.items():
        if isinstance(value, dict):
            # subset entries: substitute {version} in both keys (used by
            # X meta-protocol member names) and string values
            resolved[key] = {
                _substitute_version(k, version): _substitute_version(v, version)
                for k, v in value.items()
            }
        else:
            resolved[key] = value

    for subset, subset_overlay in overlay.items():
        if not isinstance(subset_overlay, dict):
            msg = (
                f"Version '{version}' of {where}: overlay for subset '{subset}' "
                f"must be a mapping of entry keys to values, got: {subset_overlay!r}"
            )
            raise ValueError(msg)
        base = dict(resolved.get(subset, dict()))
        for key, value in subset_overlay.items():
            base[key] = _substitute_version(value, version)
        resolved[subset] = base

    return resolved


def _expand_versions(
    db_name: Text,
    task_name: Text,
    protocol: Text,
    protocol_entries,
    database_yml,
):
    """Expand a protocol block into its per-version variants.

    Returns a list of (protocol_name, entries, version) tuples:
    - one ("<protocol>@<version>", entries, version) per declared version,
    - plus ("<protocol>", entries, version) for the default version.

    Protocol blocks without a `versions:` key are returned as
    [(protocol, entries, None)], i.e. the exact unversioned behavior.
    """

    where = f"'{db_name}.{task_name}.{protocol}' ({database_yml})"

    if not isinstance(protocol_entries, dict) or "versions" not in protocol_entries:
        if isinstance(protocol_entries, dict) and _uses_version_placeholder(
            protocol_entries
        ):
            msg = (
                f"{where} uses the {{version}} placeholder "
                f"but does not declare a 'versions:' block."
            )
            raise ValueError(msg)
        return [(protocol, protocol_entries, None)]

    entries = dict(protocol_entries)
    versions = entries.pop("versions")

    # `versions` is either a list of names (no overlays) or a mapping
    # {name: overlay}. Normalize to an ordered {name: overlay} mapping.
    if isinstance(versions, (list, tuple)):
        versions_map = {str(v): {} for v in versions}
        if len(versions_map) != len(versions):
            msg = (
                f"{where}: duplicate version name in 'versions:' list "
                f"{list(versions)}."
            )
            raise ValueError(msg)
    elif isinstance(versions, dict):
        versions_map = {str(v): overlay for v, overlay in versions.items()}
    else:
        versions_map = None

    if not versions_map:
        msg = (
            f"{where}: 'versions' must be a non-empty list of version names, "
            f"or a mapping (version name -> optional per-subset overrides)."
        )
        raise ValueError(msg)

    version_names = list(versions_map)

    # default version: explicit `version:` key, else the conventional
    # "original" version if present, else the first declared version.
    if "version" in entries:
        default = str(entries.pop("version"))
        if default not in version_names:
            msg = (
                f"{where}: default version '{default}' is not declared in "
                f"'versions:' (available: {sorted(version_names)})."
            )
            raise ValueError(msg)
    elif _DEFAULT_VERSION in version_names:
        default = _DEFAULT_VERSION
    else:
        default = version_names[0]

    expanded = []
    for version_name, overlay in versions_map.items():
        resolved = _apply_version(entries, overlay, version_name, where)
        expanded.append((f"{protocol}@{version_name}", resolved, version_name))
        if version_name == default:
            # fresh copy: create_protocol destructively pops entries
            # (e.g. "scope"), so variants must not share dicts
            resolved_again = _apply_version(entries, overlay, version_name, where)
            expanded.append((protocol, resolved_again, version_name))

    return expanded


# To ease the understanding of future me, all comments inside Registry codebase
# assume the existence of the following database.yml files.

# ======================================
# Content of /path/to/first/database.yml
# ======================================
# Databases:
#     DatabaseA:
#         - relative/path/A/trn/{uri}.wav
#         - relative/path/A/dev/{uri}.wav
#         - relative/path/A/tst/{uri}.wav
#     DatabaseB: /absolute/path/B/{uri}.wav
#
# Protocols:
#     DatabaseA:
#         SpeakerDiarization:
#             ProtocolA:
#                 train:
#                     uri: relative/path/A/trn.lst
#                 development:
#                     uri: relative/path/A/dev.lst
#                 test:
#                     uri: relative/path/A/tst.lst
#             ProtocolB:
#                 ...
#     DatabaseB:
#         SpeakerDiarization:
#             Protocol:
#                 ...
#     X:
#         SpeakerDiarization:
#             A_and_B:
#                 train: ...
#                 development: ...
#                 test: ...

# ======================================
# Content of /path/to/second/database.yml
# ======================================
# Databases:
#     DatabaseC: /absolute/path/C/{uri}.wav
#     DatabaseB: /absolute/path/B/{uri}.wav
# Protocols:
#     DatabaseB:
#         SpeakerDiarization:
#             Protocol:
#                ...
#     DatabaseC:
#         SpeakerDiarization:
#             Protocol:
#                 ...


class Registry:
    """Database and experimental protocols registry

    Usage
    -----
    >>> from pyannote.database import registry
    >>> registry.load_database("/path/to/first/database.yml")
    >>> registry.load_database("/path/to/second/database.yml")
    """

    def __init__(self) -> None:
        # Mapping of database.yml paths to their config in a dictionary
        # Example after loading both database.yml:
        #   {"/path/to/first/database.yml": {
        #           "Databases":{
        #               "DatabaseA": ["relative/path/A/trn/{uri}.wav", "relative/path/A/dev/{uri}.wav", relative/path/A/tst/{uri}.wav]
        #               "DatabaseB": "/absolute/path/B/{uri}.wav"
        #           },
        #           "Protocols":{
        #               "DatabaseA":{
        #                   "SpeakerDiarization": {
        #                       "ProtocolA": {
        #                           "train": {"uri": "relative/path/A/trn.lst"},
        #                           "development": {"uri": "relative/path/A/dev.lst"},
        #                           "test": {"uri"; "relative/path/A/tst.lst"}
        #                       }
        #                   }
        #               },
        #               "DatabaseB":{"SpeakerDiarization":{"Protocol": {...}}},
        #               "X":{"SpeakerDiarization":{"A_and_B":{...}}}
        #           }
        #       },
        #    "/path/to/second/database.yml": {
        #           "Databases":{
        #               "DatabaseC": /absolute/path/C/{uri}.wav
        #               "DatabaseB": "/absolute/path/B/{uri}.wav"
        #           },
        #           "Protocols":{
        #               "DatabaseB":{"SpeakerDiarization": {"Protocol": {...}}},
        #               "DatabaseC":{...}
        #           }
        #       }
        #   }
        self.configs: Dict[Path, Dict] = dict()

        # Content of the "Database" root item (= where to find file content)
        # Example after loading both database.yml:
        #   {
        #   "DatabaseA": [
        #       "/path/to/first/relative/path/A/trn/{uri}.wav",
        #       "/path/to/first/relative/path/A/dev/{uri}.wav",
        #       /path/to/first/relative/path/A/tst/{uri}.wav
        #       ],
        #   "DatabaseB": ["/absolute/path/B/{uri}.wav"],
        #   "DatabaseC": ["/absolute/path/C/{uri}.wav"]
        #   }
        self.sources: Dict[Text, List[Text]] = dict()

        # Mapping of database names to a type that inherits from Database
        # Example after loading both database.yml:
        #   {"DatabaseA": pyannote.database.registry.DatabaseA,
        #    "DatabaseB": pyannote.database.registry.DatabaseB,
        #    "DatabaseC": pyannote.database.registry.DatabaseC,
        #    "X": pyannote.database.registry.X}
        self.databases: Dict[Text, Type] = dict()

    def load_database(
        self,
        path: Union[Text, Path],
        mode: LoadingMode = LoadingMode.OVERRIDE,
    ):
        """Load YAML configuration file into the registry

        Parameters
        ----------
        path : str or Path
            Path to YAML configuration file.
        mode : LoadingMode, optional
            Controls how to handle conflicts in protocol names.
            Defaults to overriding the existing protocol.

        Usage
        -----
        >>> from pyannote.database import registry
        >>> registry.load_database("/path/to/database.yml")
        """

        self._load_database_helper(path, mode=mode, loading=set())
        self._reload_meta_protocols()

    def _load_database_helper(
        self,
        database_yml: Union[Text, Path],
        mode: LoadingMode = LoadingMode.KEEP,
        loading: Set[Path] = set(),
    ):
        """Helper function for recursive loading

        Parameters
        ----------
        database_yml : Union[Text, Path]
            Path to the database.yml
        mode : LoadingMode, optional
            Controls how to handle conflicts in protocol names.
            Defaults to overriding the existing protocol.
        """

        # make path absolute
        database_yml = Path(database_yml).expanduser().resolve()

        # stop here if configuration file is already being loaded
        # (possibly because of circular requirements)
        if database_yml in loading:
            return

        # mark it as currently being loaded (to avoid future circular requirements)
        loading.add(database_yml)

        # load configuration
        with open(database_yml, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        # load every requirement
        requirements = config.pop("Requirements", list())
        if not isinstance(requirements, list):
            requirements = [requirements]
        for requirement_yaml in requirements:
            requirement_yaml = Path(requirement_yaml)
            if not requirement_yaml.is_absolute():
                requirement_yaml = database_yml.parent / requirement_yaml

            self._load_database_helper(requirement_yaml, mode=mode, loading=loading)

        # process "Protocols" section
        protocols = config.get("Protocols", dict())

        # make sure meta-protocols are processed last (relies on the fact that
        # dicts are iterated in insertion order since Python 3.6)
        x = protocols.pop("X", None)
        if x is not None:
            protocols["X"] = x

        # load protocols of each database
        for db_name, db_entries in protocols.items():
            self._load_protocols(db_name, db_entries, database_yml, mode=mode)

        # process "Databases" section
        databases = config.get("Databases", dict())
        for db_name, value in databases.items():
            if not isinstance(value, list):
                value = [value]

            path_list: List[str] = list()
            for p in value:
                path = Path(p)
                if not path.is_absolute():
                    path = database_yml.parent / path
                path_list.append(str(path))
            self.sources[str(db_name)] = path_list

        # save configuration for later reloading of meta-protocols
        self.configs[database_yml] = config

    def get_database(self, database_name, **kwargs) -> Database:
        """Get database by name

        Parameters
        ----------
        database_name : str
            Database name.

        Returns
        -------
        database : Database
            Database instance
        """

        try:
            database = self.databases[database_name]

        except KeyError:
            if database_name == "X":
                msg = (
                    "Could not find any meta-protocol. Please refer to "
                    "pyannote.database documentation to learn how to define them: "
                    "https://github.com/pyannote/pyannote-database"
                )
            else:
                msg = (
                    'Could not find any protocol for "{name}" database. Please '
                    "refer to pyannote.database documentation to learn how to "
                    "define them: https://github.com/pyannote/pyannote-database"
                )
                msg = msg.format(name=database_name)
            raise ValueError(msg)

        return database(**kwargs)

    def get_protocol(
        self, name, preprocessors: Optional[Preprocessors] = None
    ) -> Protocol:
        """Get protocol by full name

        Parameters
        ----------
        name : str
            Protocol full name (e.g. "Etape.SpeakerDiarization.TV").
            An annotation version may be selected with an "@" suffix
            (e.g. "AMI.SpeakerDiarization.Raw@only_words"); without it,
            the protocol's default version is used.
        preprocessors : dict or (key, preprocessor) iterable
            When provided, each protocol item (dictionary) are preprocessed, such
            that item[key] = preprocessor(item). In case 'preprocessor' is not
            callable, it should be a string containing placeholder for item keys
            (e.g. {'audio': '/path/to/{uri}.wav'})

        Returns
        -------
        protocol : Protocol
            Protocol instance
        """

        # extract the optional "@version" suffix first so that version names
        # may contain dots (e.g. "Raw@v2.1")
        base_name, _, version = name.partition("@")
        database_name, task_name, protocol_name = base_name.split(".")
        if version:
            protocol_name = f"{protocol_name}@{version}"
        database = self.get_database(database_name)
        try:
            protocol = database.get_protocol(
                task_name, protocol_name, preprocessors=preprocessors
            )
        except KeyError:
            if not version:
                raise
            base_protocol = protocol_name.split("@", 1)[0]
            available = [
                p.split("@", 1)[1]
                for p in database.get_protocols(task_name)
                if p.startswith(f"{base_protocol}@")
            ]
            msg = (
                f"Could not find version '{version}' of protocol "
                f"'{database_name}.{task_name}.{base_protocol}' "
                f"(available versions: {sorted(available) if available else 'none'})."
            )
            raise ValueError(msg)
        protocol.name = name
        return protocol

    # iterate over all protocols by name
    def __iter__(self):
        for database_name in self.databases:
            database = self.get_database(database_name)
            for task_name in database.get_tasks():
                for protocol_name in database.get_protocols(task_name):
                    yield f"{database_name}.{task_name}.{protocol_name}"

    def _load_protocols(
        self,
        db_name,
        db_entries: dict,
        database_yml: Union[Text, Path] = None,
        mode: LoadingMode = LoadingMode.OVERRIDE,
    ):
        """Load all protocols from this database into the registry.

        Parameters
        ----------
        db_name : _type_
            Name of the database
        db_entries : dict
            Dict of all entries under this database (this should be tasks)
        database_yml : Union[Text, Path], optional
            Path to the database.yml file. Not required for X protocols, by default None
        """

        db_name = str(db_name)

        # maps tuple (task,protocol) to the custom protocol class
        protocols: Dict[Tuple[Text, Text], Type] = dict()

        for task_name, task_entries in db_entries.items():

            # `versions` / `version` may be declared once at the task level
            # (shared by every protocol of the task) instead of per protocol.
            task_versions = task_version = None
            if isinstance(task_entries, dict):
                task_versions = task_entries.get("versions")
                task_version = task_entries.get("version")

            for protocol, protocol_entries in task_entries.items():

                # reserved task-level keys, not protocols
                if protocol in ("versions", "version"):
                    continue

                protocol = str(protocol)

                # a protocol that uses the {version} placeholder but does not
                # declare its own `versions` inherits the task-level ones;
                # a protocol that uses no {version} stays unversioned even under
                # a versioned task.
                if (
                    task_versions is not None
                    and isinstance(protocol_entries, dict)
                    and "versions" not in protocol_entries
                    and _uses_version_placeholder(protocol_entries)
                ):
                    protocol_entries = dict(protocol_entries)
                    protocol_entries["versions"] = task_versions
                    if task_version is not None and "version" not in protocol_entries:
                        protocol_entries["version"] = task_version

                # expand `versions:` blocks into one protocol per version
                # (a single [(protocol, entries, None)] when unversioned)
                for variant_name, variant_entries, version in _expand_versions(
                    db_name, task_name, protocol, protocol_entries, database_yml
                ):
                    CustomProtocol = create_protocol(
                        db_name,
                        task_name,
                        variant_name,
                        variant_entries,
                        database_yml,
                        annotation_version=version,
                    )
                    if CustomProtocol is None:
                        continue

                    protocols[(task_name, variant_name)] = CustomProtocol

        # If needed, merge old protocols dict with the new one (according to current override rules)
        if db_name in self.databases:
            old_protocols = self.databases[db_name]._protocols
            _merge_protocols_inplace(
                protocols, old_protocols, mode, db_name, database_yml
            )

        # create database class on-the-fly
        protocol_list = [
            (task, p_name, p_type) for (task, p_name), p_type in protocols.items()
        ]
        self.databases[db_name] = type(
            db_name,
            (Database,),
            {"__init__": get_init(protocol_list), "_protocols": protocols},
        )

    def _reload_meta_protocols(self):
        """Reloads all meta protocols from all database.yml files loaded."""

        # TODO: decide how to handle X protocol overriding.

        self.databases.pop("X", None)

        for db_yml, config in self.configs.items():
            databases = config.get("Protocols", dict())
            if "X" in databases:
                self._load_protocols(
                    "X", databases["X"], db_yml, mode=LoadingMode.OVERRIDE
                )


def _env_config_paths() -> List[Path]:
    """Parse PYANNOTE_DATABASE_CONFIG environment variable

    PYANNOTE_DATABASE_CONFIG may contain multiple paths separation by ";".

    Returns
    -------
    paths : list of Path
        List of all YAML database file defined in PYANNOTE_DATABASE_CONF
    """

    content = os.environ.get("PYANNOTE_DATABASE_CONFIG", "")

    paths = []
    for path in content.split(";"):
        path = Path(path).expanduser()
        if path.is_file():
            paths.append(path)
    return paths


def _find_default_ymls() -> List[Path]:
    """Get paths to default YAML configuration files

    * $HOME/.pyannote/database.yml
    * $CWD/database.yml
    * PYANNOTE_DATABASE_CONFIG environment variable

    Returns
    -------
    paths : list of Path
        List of existing default YAML configuration files
    """

    paths: List[Path] = []

    home_db_yml = Path("~/.pyannote/database.yml").expanduser()
    if home_db_yml.is_file():
        paths.append(home_db_yml)

    cwd_db_yml = Path.cwd() / "database.yml"
    if cwd_db_yml.is_file():
        paths.append(cwd_db_yml)

    paths += _env_config_paths()

    return paths


def _merge_protocols_inplace(
    new_protocols: Dict[Tuple[Text, Text], Type],
    old_protocols: Dict[Tuple[Text, Text], Type],
    mode: LoadingMode,
    db_name: str,
    database_yml: str,
):
    """Merge new and old protocols inplace into the passed new_protocol.

    Warning, merging order might be counterintuitive : "KEEP" strategy keeps element from the OLD protocol
    and MODIFIES the new protocol.

    TODO: make it intuitive :)

    Parameters
    ----------
    new_protocols : Dict[Tuple[Text, Text], Type]
        New protocol dict
        Maps (task,protocol) tuples to custom protocol classes
    old_protocols : Dict[Tuple[Text, Text], Type]
        Old protocols dict
        Maps (task,protocol) tuples to custom protocol classes.
    mode : LoadingMode
        How to handle override
    db_name : _type_
        Name of the database (for logging/warning purposes)
    database_yml : str
        Path of the database.yml file (for logging/warning purposes)
    """

    # for all previously defined protocol (in old_protocols)
    for p_id, old_p in old_protocols.items():
        # if this protocol is redefined
        if p_id in new_protocols:
            t_name, p_name = p_id
            realname = f"{db_name}.{t_name}.{p_name}"

            # raise an error
            if mode == LoadingMode.ERROR:
                raise RuntimeError(
                    f"Cannot load {realname} protocol from '{database_yml}' as it already exists."
                )

            # keep the new protocol
            elif mode == LoadingMode.OVERRIDE:
                warnings.warn(
                    f"Replacing existing {realname} protocol by the one defined in '{database_yml}'."
                )
                pass

            # keep the old protocol
            elif mode == LoadingMode.KEEP:
                warnings.warn(
                    f"Skipping {realname} protocol defined in '{database_yml}' as it already exists."
                )
                new_protocols[p_id] = old_p

        # no conflit : keep the previously defined protocol
        else:
            new_protocols[p_id] = old_p


# initialize the registry singleton
registry = Registry()

# load all database yaml files found at startup
for yml in _find_default_ymls():
    registry.load_database(yml)
