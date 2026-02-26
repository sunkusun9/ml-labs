# Pipeline

::: mllabs._pipeline.Pipeline
    options:
      members:
        - set_grp
        - set_node
        - get_node_names
        - get_node_attrs
        - get_node
        - get_grp
        - rename_grp
        - remove_grp
        - remove_node
        - copy
        - copy_stage
        - copy_nodes
        - compare_nodes
        - desc_pipeline
        - desc_node

::: mllabs._pipeline.PipelineGroup
    options:
      members:
        - get_attrs
        - diff

::: mllabs._pipeline.PipelineNode
    options:
      members:
        - get_attrs
        - diff
