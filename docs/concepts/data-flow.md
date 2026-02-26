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

### In Trainer

```python
data_dict = {
    'X': (X_train, X_valid),
    'y': (y_train, y_valid),
}
```

No inner fold — data is split once by the `splitter`.

## Cache

`Experimenter` uses an LRU cache (capacity-based, default 4 GB) to store Stage outputs. When a Stage node's output is requested by multiple downstream nodes, it is computed once and reused from cache.

`Trainer` shares the same cache instance with its parent `Experimenter`, using `"train_all"` as the type key to avoid collisions.

## X-less Nodes

If a node's `edges` contains only `'y'` and no `'X'` (e.g. `LabelEncoder`), the `'y'` data is used as the primary input. The processor receives the y array directly, and its output becomes the new y columns.
