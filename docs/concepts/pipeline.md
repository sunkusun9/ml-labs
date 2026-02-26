# Pipeline

A `Pipeline` is a node graph that describes the structure of an ML workflow.

## Roles

Every node has one of three roles:

| Role | Class | Purpose |
|------|-------|---------|
| **DataSource** | *(implicit)* | The original input data. Not a real node — represented as `None` in edges. |
| **Stage** | `TransformProcessor` | Transforms data and passes it downstream. Stays alive to supply data to child nodes. |
| **Head** | `PredictProcessor` | Consumes transformed data and produces predictions. Terminal node. |

## Nodes and Groups

A **node** is the unit of execution. Each node has:

- `processor`: the class that does the work (e.g. `StandardScaler`, `LGBMClassifier`)
- `edges`: which upstream nodes supply which variables
- `method`: method name to call on the processor
- `adapter`: optional wrapper that translates data and params to framework conventions
- `params`: constructor parameters for the processor

A **group** (`PipelineGroup`) lets multiple nodes share configuration. Node attributes override group attributes; group attributes override parent group attributes.

## edges

`edges` defines what data a node receives and from where.

```python
edges = {
    'X': [(None, ['feature1', 'feature2']),   # from DataSource
           ('stage1', None)],                  # all columns from stage1
    'y': [(None, 'target')],
}
```

- Keys name variable sets (`'X'`, `'y'`, `'sample_weight'`, …)
- Each value is a list of `(node_name, var_spec)` pairs
  - `node_name=None` means DataSource
  - `var_spec`: `None` (all columns), `str`, `list`, or `callable`
- Multiple entries for the same key are concatenated column-wise
- Child nodes inherit and extend parent group edges
