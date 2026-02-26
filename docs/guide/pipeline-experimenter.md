# Pipeline & Experimenter

## Pipeline

### Defining Groups and Nodes

A `Pipeline` is built by first defining groups, then nodes inside those groups.

```python
from mllabs import Experimenter
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

exp = Experimenter(df, path='./exp', sp=StratifiedKFold(n_splits=5))
```

**Groups** (`set_grp`) define shared configuration for one or more nodes:

```python
exp.set_grp('scaler', role='stage', processor=StandardScaler,
            edges={'X': [(None, features)]}, method='fit_transform')

exp.set_grp('lgbm', role='head', processor=LGBMClassifier,
            edges={'X': [(None, features)], 'y': [(None, 'target')]},
            method='predict_proba',
            params={'n_estimators': 300, 'learning_rate': 0.05})
```

**Nodes** (`set_node`) are the executable units inside a group:

```python
exp.set_node('lgbm_v1', grp='lgbm', params={'num_leaves': 31})
exp.set_node('lgbm_v2', grp='lgbm', params={'num_leaves': 63})
```

Node parameters override group parameters. Processor, edges, method, and adapter are inherited if not specified on the node.

#### Name Restrictions

Names cannot contain `__` or any of `/ \ < > : " | ? *`.

### edges Syntax

`edges` maps variable set names to a list of `(node_name, var_spec)` pairs:

```python
edges = {
    'X': [
        (None, ['col1', 'col2']),   # specific columns from DataSource
        ('scaler_node', None),       # all output columns from a Stage node
    ],
    'y': [(None, 'target')],
    'sample_weight': [(None, 'weight')],
}
```

`node_name=None` refers to DataSource. Multiple entries for the same key are concatenated column-wise.

**var_spec options:**

| Type | Behavior |
|------|----------|
| `None` | All columns |
| `str` | Single column by name |
| `list` | Multiple columns by name |
| `callable` | Applied to available columns, returns list |

### Group Hierarchy and Attribute Inheritance

Groups can be nested using the `parent` parameter:

```python
exp.set_grp('models', role='head',
            edges={'X': [('scaler', None)], 'y': [(None, 'target')]},
            method='predict_proba')

exp.set_grp('lgbm', role='head', processor=LGBMClassifier, parent='models')
exp.set_grp('xgb',  role='head', processor=XGBClassifier,  parent='models')
```

Inheritance rules:

- **processor**, **method**, **adapter**: child overrides parent (if set)
- **edges**: child entries are prepended; parent entries appended (same key → extend)
- **params**: child wins on conflict; parent fills missing keys

### copy and compare_nodes

**Copy variants:**

```python
p2 = pipeline.copy()                          # full copy
p_stage = pipeline.copy_stage()               # stage groups/nodes only
p_sub = pipeline.copy_nodes(['node_a', 'node_b'])  # nodes + all ancestors
```

**Comparing nodes** with the same processor:

```python
diffs = exp.pipeline.compare_nodes(['lgbm_v1', 'lgbm_v2', 'lgbm_v3'])
# Returns {processor_name: DataFrame} showing only differing params and X variable sets
```

Columns with identical values across all nodes are excluded — only differences are shown.

---

## Experimenter

### Instantiation

```python
from sklearn.model_selection import StratifiedKFold, ShuffleSplit
from mllabs import Experimenter

exp = Experimenter(
    data=df,
    path='./my_exp',
    sp=StratifiedKFold(n_splits=5),   # outer splits
    sp_v=ShuffleSplit(n_splits=1),     # inner splits (optional)
    splitter_params={'y': 'target'},   # columns to pass to splitter
    title='My Experiment',
    data_key='v1',                     # verified on load
    cache_maxsize=4 * 1024**3,         # 4 GB LRU cache
)
```

Use `Experimenter.create()` instead to raise an error if the path already exists.

To reload a saved experiment:

```python
exp = Experimenter.load('./my_exp', data=df, data_key='v1')
```

### build / exp Workflow

```python
exp.build()         # builds all Stage nodes not yet built
exp.build(rebuild=True)   # rebuilds even already-built Stage nodes

exp.exp(['lgbm_v1', 'lgbm_v2'])   # runs specified Head nodes
exp.exp()                          # runs all Head nodes
```

`build()` processes Stage nodes in topological order. `exp()` first calls `build()` implicitly for any missing Stage dependencies, then runs the Head nodes.

Both methods skip already-built nodes unless `rebuild=True` (build only).

### reset_nodes, show_error_nodes

```python
exp.reset_nodes(['lgbm_v1'])   # resets to init: removes node_objs, clears cache/collectors
```

`reset_nodes` also propagates to downstream nodes and their Collectors and Trainers.

```python
exp.show_error_nodes()                   # lists all nodes in error state
exp.show_error_nodes(traceback=True)     # includes full traceback
exp.show_error_nodes(['lgbm_v1'])        # check a specific node
```

### finalize / reinitialize / close_exp / reopen_exp

**`finalize(nodes)`** — releases memory for built Head nodes (disk artifacts remain):

```python
exp.finalize(['lgbm_v1'])   # built → finalized
```

**`reinitialize(nodes)`** — removes finalized nodes from tracking (returns to init state):

```python
exp.reinitialize(['lgbm_v1'])   # finalized → init
```

**`close_exp()`** — finalizes all built nodes and marks the experiment as closed. Collector data is preserved.

```python
exp.close_exp()   # open → closed
```

**`reopen_exp()`** — clears all node objects, sets status back to open, and rebuilds Stage nodes.

```python
exp.reopen_exp()   # closed → open (+ rebuild)
```

### Adding and Using Collectors

Collectors capture data during `exp()` for each matched Head node.

```python
from mllabs.collector import MetricCollector
from sklearn.metrics import log_loss
from mllabs import Connector

collector = MetricCollector(
    name='metrics',
    connector=Connector(),        # matches all nodes
    output_var=None,              # use all output columns
    metric_func=log_loss,
    include_train=True,
)

exp.add_collector(collector)   # registers and runs collect() on existing built nodes
```

Querying results:

```python
mc = exp.get_collector('metrics')
mc.get_metric('lgbm_v1')              # per-fold metrics for one node
mc.get_metrics(['lgbm_v1', 'lgbm_v2'])
mc.get_metrics_agg(nodes=None, inner_fold='mean', outer_fold='mean')
```

**Ad-hoc collection** on already-built nodes:

```python
exp.collect(collector, exist='skip')   # skip nodes already collected
```

Removing a collector:

```python
exp.remove_collector('metrics')
```
