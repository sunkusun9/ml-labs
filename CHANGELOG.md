# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
