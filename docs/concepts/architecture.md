# Architecture

ml-labs is composed of four core modules, each with a distinct responsibility.

```
Pipeline ──────── defines the node graph (structure)
    │
Experimenter ──── executes builds and experiments (single dataset)
    │
Trainer ────────── trains with cross-validation splits
    │
Inferencer ─────── applies trained processors to new data
```

## Pipeline

`Pipeline` is a directed graph of nodes that describes the ML workflow structure — which processors exist, how they connect, and what parameters they use. It holds no data and performs no computation. It is the blueprint that `Experimenter` and `Trainer` read.

## Experimenter

`Experimenter` takes a `Pipeline` and a dataset, then executes the graph node by node. It manages:

- **Build** (`build()`): runs Stage nodes (transformers)
- **Experiment** (`exp()`): runs Head nodes (predictors)
- **Collectors**: pluggable objects that capture metrics, outputs, SHAP values, or stacking data during execution
- **Cache**: LRU cache (capacity-based) to avoid recomputing Stage outputs

## Trainer

`Trainer` handles cross-validation. It splits data using a `splitter`, then runs each node across all splits. Stage outputs are kept in memory; Head outputs are written to disk per split. The result can be converted to an `Inferencer` via `to_inferencer()`.

## Inferencer

`Inferencer` holds the fitted processors produced by `Trainer`. Given new data, it runs each split's processors and aggregates the results (`mean`, `mode`, or a custom callable).
