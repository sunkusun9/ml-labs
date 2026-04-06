# Data Flow

## Overview

Data moves through the pipeline from DataSource → Stage → Head, assembled at each node according to its `edges` definition.

```
DataSource
    │
    ▼
Stage A  ──►  Stage B
                 │
                 ▼
              Head C
```

When `Head C` runs, its `edges` are resolved by pulling columns from `DataSource` and/or the outputs of `Stage A`, `Stage B`.

## data_dict Structure

Each node receives a `data_dict` — a dict mapping edge keys to data.

### In Experimenter

```python
data_dict = {
    'X': ((X_train, X_train_v), X_valid),
    'y': ((y_train, y_train_v), y_valid),
}
```

- `train`: full training fold data
- `train_v`: training fold filtered by `output_var` (for inner validation)
- `valid`: validation fold data

## Cache

`Experimenter` uses an LRU cache (capacity-based, default 4 GB) to store Stage outputs. When a Stage node's output is requested by multiple downstream nodes, it is computed once and reused from cache.

`Trainer` shares the same cache instance with its parent `Experimenter`. To avoid key collisions, each `TrainFold` uses a negative `outer_idx` (`-(split_idx + 1)`), which never overlaps with the Experimenter's positive fold indices.

## Inference Data Flow

At inference time, `Inferencer` builds an `InferenceDataFlow` per split. This is a lightweight in-memory graph that holds only the fitted processors — no disk or cache dependency. It resolves only `'X'` edges; `'y'` and `'sample_weight'` edges are training-only and are ignored during inference.

## X-less Nodes

If a node's `edges` contains only `'y'` and no `'X'` (e.g. `LabelEncoder`), the `'y'` data is used as the primary input. The processor receives the y array directly, and its output becomes the new y columns.
