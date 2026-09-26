# TrawMem

This repository contains the TrawMem implementation described in the paper:
**Task-Oriented Role-Adaptive Workspaces**.

TrawMem builds persistent interaction threads and a sparse thread graph. For a
query, it routes to local threads and assembles a temporary workspace using
query-local Anchor, Bridge, and Context roles. Role assignments are discarded
after workspace construction and are never stored as permanent memory labels.

The public Python package is `trawmem`, and benchmark adapters import it
directly. The method and result names exposed by this repository use `TrawMem`.

## Layout

- `trawmem/`: implementation of thread construction, graph routing, role-aware workspace assembly, and answer generation
- `tests/`: focused unit tests
- `test_locomo10.py`: LoCoMo evaluation adapter
- `test_memgallery.py`: MemGallery evaluation adapter
- `config.py.example`: configuration template; copy it to `config.py` locally

## Install

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

Set model and API settings through `config.py` or environment variables. Keep
`config.py`, datasets, model weights, caches, and generated outputs outside the
repository. Never commit credentials.

## Verify

```bash
PYTHONPATH=. python3 tests/test_thread_workspace.py
```

The benchmark runner expects paths supplied by the caller:

```bash
TRAWMEM_DATASET_PATH=/path/to/locomo10.json \
TRAWMEM_MODEL_PATH=/path/to/model \
TRAWMEM_EMBEDDING_PATH=/path/to/embedding-model \
/bin/bash ./run-trawmem.sh
```

Use `TRAWMEM_BENCHMARK=memgallery` for the MemGallery adapter. The runner
writes logs and results to the configured output directory, which is ignored by
Git. The paper evaluation scope is LoCoMo and MemGallery.

The full benchmark wrapper is `run-trawmem-full.sh`; provide dataset paths
through `TRAWMEM_LOCOMO_DATASET` and `TRAWMEM_MEMGALLERY_DATASET`.

## Reproducibility scope

The repository contains source code, tests, configuration templates, and the
MIT license. It does not contain paper files, private paths, datasets,
model weights, API keys, experiment outputs, caches, or generated figures.
