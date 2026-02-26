# Collectors

::: mllabs.collector.Collector
    options:
      members:
        - has
        - has_node
        - reset_nodes
        - save
        - load

::: mllabs.collector.MetricCollector
    options:
      members:
        - get_metric
        - get_metrics
        - get_metrics_agg

::: mllabs.collector.StackingCollector
    options:
      members:
        - get_dataset

::: mllabs.collector.ModelAttrCollector
    options:
      members:
        - get_attr
        - get_attrs
        - get_attrs_agg

::: mllabs.collector.SHAPCollector
    options:
      members:
        - get_feature_importance
        - get_feature_importance_agg

::: mllabs.collector.OutputCollector
    options:
      members:
        - get_output
        - get_outputs
