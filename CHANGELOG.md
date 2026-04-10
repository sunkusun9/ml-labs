# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-04-10

### Added

- `DefaultLogger`: new ANSI cursor-movement multi-session progress display — each session occupies one terminal line, redrawn in-place; falls back to plain `print` in non-TTY environments (Jupyter, pipes)
- `ProgressSessionLogger`: renamed from the old `DefaultLogger`; session_cls injection pattern preserved (`TqdmProgressSession` etc.)
- `MetricCollector`, `ModelAttrCollector`, `SHAPCollector`: ad-hoc collection without path — when `path=None`, results accumulate in `_cache` only; all query methods (`get_metric`, `get_attr`, `get_feature_importance`, etc.) work against the cache
- `Experimenter.collect`: calls `_setup(n_outer, n_inner)` before ad-hoc loop so `_flush_outer` triggers correctly for path=None collectors
- `Experimenter.add_collector`: `exist='replace'` mode added
- `NNClassifier`/`NNRegressor`: `device` parameter for explicit GPU assignment (issue #103)
- GPU device profiling and injection interface added to adapters (issue #99)

### Changed

- Collector/Executor architecture fully redesigned for parallel execution (issue #107)
  - Node storage split into `_store.py`, `_flow.py`, `_tracker.py`
  - `DataFlow`/`TrainDataFlow`: encapsulate per-fold data assembly
  - `ArtifactStore`: per-fold artifact read/write abstraction
  - `ExecuteTracker`: progress and state tracking for build/exp runs
  - Collector interface switched to push-based model
- Head node management refactored from `HeadObj` to function-based API (issue #105)
- `Trainer`/`Inferencer` modernized with `TrainDataFlow`/`InferenceDataFlow`

### Fixed

- `SHAPCollector.collect`: second data filter was overwriting `train_data` with filtered `valid_data`; return block was using `train_data['X']` for both train and valid SHAP values
- `Experimenter.remove_collector`: now deletes the collector directory from disk

## [0.6.4] - 2026-03-18

### Added

- `ProcessCollector` (`mllabs/collector/_process.py`): collects predictions on external (test) data during `exp()`
  - Passes ext data through upstream Stage processors via `Experimenter.process_ext()` per outer fold
  - Inner-fold predictions aggregated by `method` (`mean`/`mode`/`simple`); outer-fold aggregated on query
  - `get_output(nodes=None, agg='mean')`: multi-node support with nodes filter (None/list/regex) and column-wise concat
  - Disk-based storage; `save`/`load` roundtrip (ext_data and experimenter not persisted)
- `Experimenter.process_ext(data, node, idx)`: passes external data through upstream Stage fitted processors for a given outer fold, yielding assembled input per inner split
- `Connector(role=...)`: new `role` parameter (`'head'`/`'stage'`) for role-based node filtering; `None` (default) skips the check
- `Pipeline.get_node_attrs`: expose `role` in returned attrs dict

### Fixed

- `Experimenter.load`: prevent `__exp.pkl` corruption when load fails — `__init__` now accepts `_save=False` to skip the initial save; `load()` passes `_save=False`
- `Experimenter.load`: add `OutputCollector` and `ProcessCollector` to `COLLECTOR_TYPES` lookup

## [0.6.3] - 2026-03-14

### Fixed

- `ModelAdapter`: add `get_process_data(data)` — adapters can override to control input type conversion in `process()`; `TransformProcessor` / `PredictProcessor` now use this instead of bare `unwrap()`
- `LightGBMAdapter`: override `get_process_data()` with polars→pandas conversion (mirrors `get_fit_params` behavior)
- `CatBoostAdapter`: version-based polars support (`>=1.3.0`); apply polars→pandas in both `get_fit_params` and `get_process_data` for older versions
- `PolarsWrapper.get_columns()`: handle `pl.Series` correctly (return `.name` instead of `.columns`)
- `TransformProcessor`: handle non-iterable `y_columns` and `str` `result.columns` in `fit_process` / `process`
- `PredictProcessor`: handle non-iterable `y_columns` in `fit`
- `StackingCollector`: wrap `str` `target_columns` in list; use `_data_cls` for `simple` / `mean` / `mode` aggregation

## [0.6.2] - 2026-03-08

### Fixed

- Pipeline: `set_grp()` now recursively invalidates child group attrs cache (`_cascade_clear_attrs`) and node attrs cache — prevents stale adapter/params being resolved when a parent group is updated
- Inferencer: `_get_process_data()` skips `resolve_columns` for DataSource edges (`src_node=None`); uses `var` directly as column spec

## [0.6.1] - 2026-03-07

### Added

- Processor: `TypeConverter(to)` — converts all columns to a target dtype (`'str'`, `'int'`, `'float'`); supports pandas, polars, and numpy
- Pipeline: `desc` attribute on `PipelineGroup` and `PipelineNode` for free-text annotations
  - Not inherited via `get_attrs`, not compared in `diff()` — desc-only changes do not trigger rebuilds
  - Updated silently on `exist='diff'` skip path

### Fixed

- LightGBM adapter: accept `early_stopping` as a plain dict of kwargs; adapter constructs `lgb_early_stopping` internally, eliminating false param-change detection
- Experimenter: collector lifecycle errors (`_start`, `_collect`, `_end_idx`, `_end`) are now caught and stored as warnings instead of propagating exceptions

## [0.6.0] - 2026-03-05

### Added

- `mllabs.nn`: sklearn-compatible neural network estimators (`NNClassifier`, `NNRegressor`) with automatic categorical embedding support
  - Auto-detects categorical columns from pandas `Categorical` / polars `Categorical` dtype
  - Auto-computes embedding dimensions `max(1, min(50, (cardinality+1)//2))`; per-column override via `embedding_dims` dict
  - Modular components: `SimpleConcatHead`, `DenseHidden`, `LogitOutput`, `BinaryLogitOutput`, `RegressionOutput`
  - `hidden` parameter accepts a dict of `DenseHidden` constructor kwargs
  - `fit(X, y, eval_set=None, callbacks=None)` with constructor callbacks and fit callbacks merged; early stopping auto-appended
  - Pickle support via `__getstate__` / `__setstate__` — weights saved only, architecture rebuilt from `col_info_` on load
- `NNAdapter` (`mllabs.adapter`): ml-labs adapter for `NNClassifier` / `NNRegressor`
  - Passes inner-validation fold as `eval_set`
  - Epoch-based progress logging via `_ProgressCallback`
  - `evals_result` exposed as `result_obj` for `ModelAttrCollector`
- `pyproject.toml`: `tensorflow` optional dependency (`pip install ml-labs[tensorflow]`)
- Experimenter: `collect()` accepts `nodes` parameter to limit collection scope
- Docs: nn module user guide (`guide/nn.md`) and API reference (`reference/nn.md`)
- Docs: Concepts index page

### Fixed

- `mllabs.nn._estimator`: guard `import tensorflow` with try/except so tf-free environments can import the package

## [0.5.0] - 2026-02-27

### Added

- Documentation: full MkDocs-based site (Material theme) published to GitHub Pages
  - Concepts: architecture, pipeline, state model, data flow
  - User guides: Pipeline & Experimenter, Trainer & Collectors, Adapters, Processors
  - Serving guide: Inferencer export, save/load, inference
  - API reference: all public classes auto-generated from docstrings via mkdocstrings

### Fixed

- `pyproject.toml`: add `README.md`, fix package description typos

## [0.4.0] - 2026-02-26

### Added
- Processor: `FrequencyEncoder`, `ColSelector`, improved `CatPairCombiner` / `CatConverter` / `CatOOVFilter`
- Processor: X-less `TransformProcessor` support for 1D transformers (e.g. `LabelEncoder`)
- SHAPCollector: `get_feature_importance`, `get_feature_importance_agg` analysis methods
- Pipeline: `exist='diff'` mode for `set_grp` / `set_node` (now default)
- Experimenter: `get_collector`, `remove_collector`, `get_trainer`, `remove_trainer`
- Logger: `rename_progress(title)` method to `BaseLogger` / `DefaultLogger`
- Logger: trailing character cleanup on progress line overwrite
- Examples: Kaggle Playground S6E2 end-to-end notebooks (EDA, feature engineering, modeling)

### Fixed
- Experimenter: `build` `rebuild` parameter not working
- Experimenter: `reopen_exp` losing collector data after `close_exp`
- Experimenter: `close_exp` not persisting status on save
- Experimenter: `exp` error handling not propagating correctly
- Pipeline: `set_grp` not collecting affected nodes when only edges change
- Adapter: safe recursive params comparison and `__eq__` for diff mode
- Processor: `CategoricalPairCombiner` output dtype to Categorical
- ExpObj: error state not persisted to disk (`error.txt`)
- Experimenter: parameter order normalized to `(node, idx)` in `get_node_output` family
- Inferencer: `process` now returns native pandas/numpy (unwrapped)

### Refactoring
- Experimenter: replace error stack trace logging with `show_error_nodes`
- Experimenter: encapsulate internal `collectors` / `trainers` dicts behind accessor methods

## [0.3.0] - 2026-02-20

### Added
- Inferencer: apply trained pipelines to new data with automatic split aggregation
  - `Trainer.to_inferencer(v)` extracts trained Processors into a standalone Inferencer
  - `process(data, agg)` supports mean/mode/callable/None aggregation
  - Single-file save/load, fully independent of Trainer

### Fixed
- StackingCollector: fix `get_dataset` shape mismatch error
- StackingCollector: preserve target data type in `get_dataset`
- MetricCollector: fix FutureWarning by adding `future_stack=True`

### Changed
- StackingCollector: add `experimenter` to constructor, remove `include_target`
- StackingCollector: remove `experimenter` from `get_dataset`, add `include_target`
- StackingCollector: pre-build index and target at construction time, remove `_build_sort_order`

## [0.1.0] - 2026-02-12

### Added
- Pipeline: DAG-based node graph management with groups and edges
- Experimenter: experiment execution engine with LRU caching and state management
- Trainer: cross-validation training pipeline with split management
- ExpObj/TrainObj: per-node build and experiment object lifecycle
- Collectors: MetricCollector, StackingCollector, ModelAttrCollector, SHAPCollector, OutputCollector
- Adapters: scikit-learn, XGBoost, LightGBM, CatBoost, Keras
- Processors: categorical encoding, imputation, pandas/polars utilities
- Data support: pandas, polars, cuDF, NumPy via DataWrapper
- Connector: flexible node matching with regex, edges, and processor filters
- Filters: DataFilter, RandomFilter, IndexFilter for data sampling
