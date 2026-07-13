# Toy example — annotation versioning

Run it (from the repo root, with the package installed):

```bash
python examples/annotation_versions/demo.py
```

Expected output:

```
registered protocols: ['Diarization', 'Diarization@original', 'Diarization@refined']

TOY.SpeakerDiarization.Diarization
    uri=a  version='original'  turns=[(0.0, 2.0, 'spk1')]
    uri=b  version='original'  turns=[(0.0, 1.0, 'spk2')]

TOY.SpeakerDiarization.Diarization@original
    uri=a  version='original'  turns=[(0.0, 2.0, 'spk1')]
    uri=b  version='original'  turns=[(0.0, 1.0, 'spk2')]

TOY.SpeakerDiarization.Diarization@refined
    uri=a  version='refined'  turns=[(0.0, 1.0, 'spk1'), (1.5, 3.0, 'spk1')]

unknown version -> Could not find version 'nope' of protocol ... (available versions: ['original', 'refined']).
```

## What it shows

- **Task-level versions** — `versions: [original, refined]` is declared once under
  `SpeakerDiarization:` and shared by every protocol of the task (here just `Diarization`),
  which becomes `Diarization`, `Diarization@original`, `Diarization@refined`.
- **`{version}` placeholder** — every differing path is templated, so each version reads from
  its own folder: `lists/{version}/train.lst` (2 uris vs 1) and `rttm/{version}/{uri}.rttm`
  (different labels). Shared files (`uem/`) omit `{version}`.
- **Default version** — bare `Diarization` uses `original` (the conventional default when no
  explicit `version:` is set).
- **`annotation_version`** — present on every file.
- **Clear error** — an unknown `@version` lists the available ones.

## Things to try

- Edit `annotations/rttm/refined/a.rttm` and re-run — the `@refined` turns change, the
  `@original` turns don't.
- Add `annotations/rttm/refined/b.rttm` and add `b` to `annotations/lists/refined/train.lst`
  — `@refined` now yields two files.
- Point `annotation` at a missing version folder and re-run — it raises `FileNotFoundError`
  (no silent fallback to the default).
- For a version that must reuse an *existing* fixed filename instead of a `{version}/` folder,
  switch `versions:` to the mapping form and add a per-subset overlay, e.g.
  `versions: {original: {}, refined: {train: {uri: annotations/lists/train.sdm.lst}}}`.

## Layout

```
annotation_versions/
├── database.yml
├── demo.py
└── annotations/
    ├── lists/
    │   ├── original/train.lst   # a, b
    │   └── refined/train.lst    # a
    ├── uem/{a.uem, b.uem}        # shared across versions
    └── rttm/
        ├── original/{a.rttm, b.rttm}
        └── refined/{a.rttm}
```
