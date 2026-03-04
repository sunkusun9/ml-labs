# ml-labs

A structured machine learning experimentation framework for building, managing, and evaluating ML pipelines with cross-validation, caching, and multi-framework support.

## Installation

```bash
pip install ml-labs
```

With optional dependencies:

```bash
pip install ml-labs[xgboost]    # XGBoost support
pip install ml-labs[lightgbm]   # LightGBM support
pip install ml-labs[catboost]   # CatBoost support
pip install ml-labs[shap]       # SHAP value analysis
pip install ml-labs[polars]      # Polars DataFrame support
pip install ml-labs[tensorflow]  # Neural network estimators (NNClassifier, NNRegressor)
pip install ml-labs[all]         # All optional dependencies
```

## Key Features

- **Pipeline**: DAG-based node graph for defining ML workflows with stages (data transformation) and heads (model prediction)
- **Experimenter**: Experiment execution engine with LRU caching, state management, and error resilience
- **Trainer**: Cross-validation training pipeline with split management
- **Collectors**: Extensible data collection — metrics, stacking outputs, model attributes, SHAP values, raw outputs
- **Adapters**: Unified interface for scikit-learn, XGBoost, LightGBM, CatBoost, and Keras
- **Data Flexibility**: Support for pandas, polars, cuDF, and NumPy arrays

## Architecture Overview

```
Pipeline          Define node graphs (stages + heads) with groups and edges
    │
Experimenter      Execute pipelines, manage cache and state
    │
  ├── ExpObj      Per-node build/experiment objects (StageObj, HeadObj)
  ├── Trainer     Cross-validation training with split management
  └── Collector   Collect metrics, predictions, model attributes, SHAP values
```

**Node State Model:**
```
init ──→ built ──→ finalized
  │
  └──→ error ──→ (reset) ──→ init
```

## Quick Start

```python
from mllabs import Experimenter, Connector, MetricCollector

exp = Experimenter(data=df, path="exp/my_experiment")

p = exp.pipeline
p.set_grp("scale", role="stage", processor="StandardScaler")
p.set_grp("model", role="head", processor="LogisticRegression",
          parent="scale", edges={"X": [(None, None)], "y": [(None, "target")]})

p.set_node("lr_default", grp="model")
p.set_node("lr_c01", grp="model", params={"C": 0.1})

mc = MetricCollector("accuracy", Connector(), output_var="prediction",
                     metric_func=lambda y, pred: (y == pred).mean())
exp.add_collector(mc)

exp.build(["lr_default", "lr_c01"])
exp.exp(["lr_default", "lr_c01"])

print(mc.get_metrics(["lr_default", "lr_c01"]))
```

## Documentation

Full documentation is available at **https://sunkusun9.github.io/ml-labs/**

- [Concepts](https://sunkusun9.github.io/ml-labs/concepts/architecture/) — Architecture, Pipeline, State model, Data flow
- [User Guide](https://sunkusun9.github.io/ml-labs/guide/pipeline-experimenter/) — Pipeline & Experimenter, Trainer & Collectors, Adapters, Processors, Neural Networks
- [Serving Guide](https://sunkusun9.github.io/ml-labs/serving/inferencer/) — Inferencer export and inference
- [API Reference](https://sunkusun9.github.io/ml-labs/reference/index/) — Full API reference

## Requirements

- Python >= 3.10
- pandas >= 1.5
- numpy >= 1.23
- scikit-learn >= 1.2
- cachetools >= 5.0

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) — free for non-commercial use.
