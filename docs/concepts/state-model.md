# State Model

## Node States (4-State)

Each node in an `Experimenter` or `Trainer` tracks its own state.

```
init ──► built ──► finalized
  ▲
  │
error ──► (reset) ──► init
```

| State | Disk | Memory | Description |
|-------|------|--------|-------------|
| **init** | — | — | Defined in Pipeline, not yet executed |
| **built** | ✓ | Stage: ✓ / Head: ✗ | Execution complete; results are accessible |
| **finalized** | ✗ | ✗ | Results extracted, resources released (Head only) |
| **error** | — | error info | Exception occurred; error details are preserved |

**Stage nodes cannot be finalized** — they must remain available to supply data to downstream nodes.

If an upstream Stage is in `error` state, downstream nodes naturally fail without any explicit propagation logic.

## State Transitions

| Method | Transition |
|--------|------------|
| `build(nodes)` | `init → built` (Stage) |
| `exp(nodes)` | `init → built` (Head) |
| `finalize()` | `built → finalized` (Head only) |
| `reset_nodes(nodes)` | any → `init` |

## Experiment States (2-State)

At the `Experimenter` level, an experiment session is either **open** or **closed**.

```
open ──► closed
```

| State | Stage objects | Collector data |
|-------|--------------|----------------|
| **open** | Kept in memory | Kept |
| **closed** | Released | Kept |

Call `close_exp()` to transition from open to closed. This releases all Stage processor objects while preserving any data collected by Collectors.
