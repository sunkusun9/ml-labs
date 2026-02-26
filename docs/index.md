# ml-labs

ML pipeline and experiment management library.

## Overview

ml-labs provides a structured framework for building, managing, and deploying machine learning pipelines.

- **Pipeline**: Node graph data structure for separating ML concerns
- **Experimenter**: Experiment execution and management
- **Trainer**: Training with cross-validation splits
- **Inferencer**: Apply trained pipelines to new data

## Installation

```bash
pip install ml-labs
```

Optional dependencies:

```bash
pip install ml-labs[xgboost]
pip install ml-labs[lightgbm]
pip install ml-labs[all]
```
