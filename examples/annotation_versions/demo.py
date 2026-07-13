#!/usr/bin/env python
"""Toy demo of annotation versioning. Run:  python examples/annotation_versions/demo.py"""

from pathlib import Path

from pyannote.database.registry import Registry

HERE = Path(__file__).resolve().parent


def turns(file):
    ann = file["annotation"]
    return [
        (round(s.start, 2), round(s.end, 2), label)
        for s, _, label in ann.itertracks(yield_label=True)
    ]


def show(registry, name):
    protocol = registry.get_protocol(name)
    print(f"\n{name}")
    for file in protocol.train():
        print(f"    uri={file['uri']}  "
              f"version={file['annotation_version']!r}  "
              f"turns={turns(file)}")


def main():
    registry = Registry()  # isolated registry (does not touch your global config)
    registry.load_database(HERE / "database.yml")

    print("registered protocols:",
          registry.get_database("TOY").get_protocols("SpeakerDiarization"))

    # default version (original): uris a + b, single-turn labels
    show(registry, "TOY.SpeakerDiarization.Diarization")

    # explicit version, same as default
    show(registry, "TOY.SpeakerDiarization.Diarization@original")

    # refined version: overlay drops uri b from train, and a's labels differ
    show(registry, "TOY.SpeakerDiarization.Diarization@refined")

    # unknown version -> a clear error naming the available versions
    try:
        registry.get_protocol("TOY.SpeakerDiarization.Diarization@nope")
    except ValueError as exc:
        print(f"\nunknown version -> {exc}")


if __name__ == "__main__":
    main()
