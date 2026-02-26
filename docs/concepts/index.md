# Concepts

This section explains the core ideas behind ml-labs — how it models ML workflows, manages state, and moves data through a pipeline.

---

## [Architecture](architecture.md)

ml-labs is built around four cooperating modules: **Pipeline**, **Experimenter**, **Trainer**, and **Inferencer**. Each has a single, well-defined responsibility. Understanding how they relate to each other is the starting point for everything else.

---

## [Pipeline](pipeline.md)

A `Pipeline` is a directed node graph that describes *what* to run and *how* nodes connect — not *when* or *with what data*. Nodes are organised into **groups** for shared configuration, and **edges** specify which upstream outputs feed into each node.

---

## [State Model](state-model.md)

Every node tracks its own lifecycle: `init → built → finalized` (or `error`). At the `Experimenter` level, a session is either **open** or **closed**. Knowing the state model helps you understand when results are available, when resources are held, and how to recover from errors.

---

## [Data Flow](data-flow.md)

Data travels from **DataSource** through **Stage** nodes to **Head** nodes, assembled at each step according to the node's `edges`. This page explains the `data_dict` structure that processors receive, how the LRU cache works, and how X-less nodes (e.g. `LabelEncoder`) are handled.
