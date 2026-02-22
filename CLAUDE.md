# CLAUDE 동작
니가 구사한 코드는 왠만한 건 다 파악 가능해. 주석은 나중에 한꺼번에 만들꺼야, 만들지마
함수나 메소드 가이드도 나중에 할꺼야.

CLAUDE.md에서 불필요하게 토큰을 낭비 하지 않도록, 작업 내역의 개요를 확인해라
작업 관리는 GitHub Issues로 한다. TODO.md 같은 파일은 만들지 마라.
Git 관련 내용(커밋 메시지, PR, 이슈 코멘트)은 영어로 작성한다.
커밋 메시지에 "Co-Authored-By" 넣지 마라. PR에 "Generated with Claude Code" 같은 광고성 메시지 넣지 마라.

# modeler 모듈 요약

## 아키텍처 개요
- **Pipeline** (`_pipeline.py`): 노드 그래프 자료구조 (ML 관심사 분리)
- **Experimenter** (`_experimenter.py`): 실험 실행/관리 (Pipeline 사용)
- **ExpObj** (`_expobj.py`): 노드별 빌드/실험 객체 관리
- **Trainer** (`_trainer.py`): 학습 실행/관리 (split 기반)
- **Inferencer** (`_inferencer.py`): 학습된 파이프라인을 새 데이터에 적용

## Node/Experiment 상태 모델

### Node 4-State
`init → built → finalized` / `init → error → (reset) → init`

| 상태 | Disk | Memory | 설명 |
|------|------|--------|------|
| **init** | - | - | Pipeline에 정의만 된 상태 |
| **built** | O | Stage: O / Head: X | 빌드 완료, 결과 추출 가능 |
| **finalized** | X | X | 결과 추출 완료, 리소스 해제 (Head 전용) |
| **error** | - | 에러 정보 | 빌드/실험 중 에러 발생, 내역 보존 |

- Stage는 finalize 불가 (하위 노드에 데이터 지속 공급)
- 상위 Stage가 error면 하위 노드도 자연스럽게 error (별도 전파 로직 불필요)

### Experiment 2-State
`open → closed`
- **open**: Stage/Head 객체 유지, Collector 데이터 유지
- **closed**: `close_exp` 호출 → Stage 객체까지 일괄 정리, Collector 데이터는 잔존

## 핵심 클래스

### Node 역할
- **DataSource** (None): 원본 데이터 제공, Pipeline에 명시적 노드 없음
- **Stage**: 전처리/변환 (TransformProcessor) — 하위 노드에 데이터 공급
- **Head**: 모델링/예측 (PredictProcessor) — 최종 결과 생산

### Pipeline 계층 (`_pipeline.py`)
- **`_params_equal(a, b)`**: params dict 안전 비교 헬퍼
  - dict → key별 재귀 비교
  - `__dict__` 있는 객체 → `__dict__` 재귀 비교 (lgb_early_stopping 등 콜백 처리)
  - `__dict__` 없는 객체(primitive, C-ext) → `==` fallback (try/except)
  - 같은 타입이어야 equal 가능, 다른 타입 → False
- **Pipeline**: 노드 그래프 관리
  - `nodes`: `{name: PipelineNode}` (None=DataSource), `grps`: `{name: PipelineGroup}`
  - `set_grp(exist='diff'|'skip'|'error'|'replace')`, `set_node(exist=...)`, `rename_grp`, `remove_grp`, `remove_node`
  - `get_node_names(query)`, `get_node_attrs(name)`, `_get_affected_nodes(nodes)`
  - `copy()`, `copy_stage()`, `copy_nodes(node_names)` — 선택적 복사
  - `compare_nodes(nodes)` → `{processor_name: DataFrame}` (params 차이 + edges['X'] stage별 변수 차이)

- **PipelineGroup**: 노드 그룹 (stage/head 역할)
  - 속성: `name`, `role`, `processor`, `edges`, `method`, `parent`, `adapter`, `params`
  - `children`: 자식 그룹명 리스트, `nodes`: 소속 노드명 리스트
  - `get_attrs(grps)`: 상위 그룹 속성 병합하여 반환
  - `diff(processor, edges, method, parent, adapter, params)`: 달라진 필드명 리스트 반환

- **PipelineNode**: 개별 노드
  - 속성: `name`, `grp`, `processor`, `edges`, `method`, `adapter`, `params`
  - `output_edges`: 이 노드를 입력으로 사용하는 노드명 리스트
  - `get_attrs(grps)`: 그룹 속성과 노드 속성 병합
  - `diff(grp, processor, edges, method, adapter, params)`: 달라진 필드명 리스트 반환

### Experimenter (`_experimenter.py`)
- 생성자: `(data, path, ..., cache_maxsize=4GB, logger)`
- `pipeline`: Pipeline 인스턴스
- `node_objs`: `{node_name: StageObj|HeadObj}`
- `cache`: DataCache (LRU, 용량 기반)
- 실행: `build(nodes, rebuild=False)` (stage), `exp(nodes)` (head)
- `close_exp()`: open→closed 전환, Stage 객체 일괄 정리 (Collector 데이터 유지)
- 상태관리: `reset_nodes(nodes)` - node_objs, cache, collectors 초기화
- 에러 조회: `show_error_nodes(nodes=None, traceback=False)` - error 상태 노드 출력
- `add_collector(collector)`: Collector 등록 (path 설정, save)
- `collect(collector, exist='skip')`: ad-hoc 수집 (빌드 완료된 head 노드 대상, progress 포함)
- 저장/로드: `_save()`, `load(filepath, data, data_key)`
  - pipeline, node_obj_keys, collector_keys 저장/복원

### DataCache (`_experimenter.py`)
- `cachetools.LRUCache` 기반, 용량(bytes) 단위 관리
- `get_data(node, typ, idx)`, `put_data(node, typ, idx, data)`
- `clear_nodes(nodes)`: 특정 노드들의 캐시 삭제

### ExpObj (`_expobj.py`)
- **StageObj**: stage 역할 노드의 빌드 객체
  - `status`: None(init) / 'built' / 'finalized' / 'error'
  - `error`: 에러 정보 dict `{type, message, traceback, fold}` (error 상태 시)
  - `load()`, `start_build()`, `build_idx()`, `end_build()`, `get_objs(idx)`, `finalize()`

- **HeadObj**: head 역할 노드의 실험 객체
  - `status`: None(init) / 'built' / 'finalized' / 'error'
  - `error`: 에러 정보 dict (error 상태 시)
  - `load()`, `start_exp()`, `exp_idx()`, `end_exp()`, `get_objs(idx)`, `finalize()`

- **에러 처리**: build/exp 중 노드별 try/except, error 시 나머지 노드 계속 진행
- **load() 버그 픽스**: 파일 없이 error 상태인 노드는 'finalized'가 아닌 'error'로 복원

### Trainer (`_trainer.py`)
- 생성자: `(name, pipeline, data, path, splitter, splitter_params, cache, logger)`
- `split_indices`: 생성자에서 `_make_splits()` 호출하여 생성. `splitter=None`이면 `None` (전체 데이터, split 없음)
- `selected_stages`, `selected_heads`: `select_head(nodes)`로 설정
- `node_objs`: `{node_name: TrainStageObj|TrainHeadObj}`
- `cache`: Experimenter에서 전달받은 DataCache 공유 (type key: `"train_all"`)
- `select_head(nodes)`: head 노드 지정 + upstream stage 자동 수집, `_get_affected_nodes`로 순서 정렬
- `train()`: 미빌드 노드만 대상, 노드별 전체 split 처리 후 다음 노드로 진행
- `process(data, v=None)`: generator, split마다 head output을 v로 필터 후 concat하여 yield
- `to_inferencer(v=None)`: 학습된 Processor를 추출하여 Inferencer 생성
- `reset_nodes(nodes)`: 하위 종속 노드 포함 초기화
- 저장/로드: `save()`, `_load(path, pipeline, data, cache, logger)`

### Inferencer (`_inferencer.py`)
- 생성자: `(pipeline, selected_stages, selected_heads, n_splits, node_objs, v=None)`
- `node_objs`: `{name: [processor_split0, processor_split1, ...]}` — Processor 리스트 (Trainer 독립)
- `process(data, agg='mean')`: split 결과 자동 집계
  - `agg`: `'mean'`/`'mode'`/callable/`None`(list 반환). 단일 split이면 집계 없이 반환
- 저장/로드: `save(path)`, `load(cls, path)` — 단일 `__inferencer.pkl`에 node_objs 포함

### TrainObj (`_trainobj.py`)
- `_train_build(node_attrs, data_dict, logger)`: Processor 생성 → fit/fit_process → `(obj, result, info)` 반환
- **TrainStageObj**: stage 노드용, `objs_` dict에 메모리 보관
  - `get_obj()`: generator, split 순서대로 `(obj, result, info)` yield
  - 파일: `obj{split_idx}.pkl`
- **TrainHeadObj**: head 노드용, 디스크에서 lazy load
  - `get_obj()`: generator, 파일에서 순차 로드하여 yield
  - 파일: `obj{split_idx}.pkl`
- Trainer용 `data_dict`: `{key: (train, valid)}` (Experimenter의 `((train, train_v), valid)`과 다름)

### Connector (`_connector.py`)
- `__init__(node_query=None, edges=None, processor=None)` — 3요소 선택적 매칭
- `match(node_name, node_attrs)`: 설정된 요소만 검사, 모두 충족 시 True
  - node_query: str(regex) 또는 list(in), edges: contain 기반 매칭, processor: 일치 검사

### Collector (`collector/` 패키지)
- **Collector** (`_base.py`): 기본 클래스
  - `__init__(name, connector)`, `path`는 add_collector 시 설정
  - 라이프사이클: `_start(node)`, `_collect(node, idx, inner_idx, context)`, `_end_idx(node, idx)`, `_end(node)`
  - `has(node)`: 수집 결과 보유 여부 (has_node에 위임)
  - `has_node(node)`, `reset_nodes(nodes)`, `save()`, `load(cls, path)`
  - `_get_nodes(nodes, available)`: None/list/str(regex) 패턴 매칭
  - context: `{node_attrs, processor, spec, input, output_train, output_valid}`

- **MetricCollector** (`_metric.py`): 메트릭 수집
  - `output_var`, `metric_func`, `include_train`
  - target: `context['input']['y']`, 예측값: `resolve_columns(output_valid, output_var)`
  - 쿼리: `get_metric(node)`, `get_metrics(nodes)`, `get_metrics_agg(nodes, inner_fold, outer_fold, include_std)`

- **StackingCollector** (`_stacking.py`): 스태킹 데이터 수집
  - `__init__(name, connector, output_var, experimenter, method='mean')`
  - 생성 시 `experimenter`에서 `_index`, `_target`(ndarray), `_target_columns` 구축
  - `output_var`, `method`(mean/mode/simple)
  - path 있으면 파일 저장, 없으면 `_mem_data`에 메모리 저장
  - 쿼리: `get_dataset(nodes=None, include_target=True)` — experimenter 불필요

- **ModelAttrCollector** (`_model_attr.py`): 모델 속성 수집 (feature_importances 등)
  - `result_key`, `adapter`(default=None, `get_adapter(connector.processor)`로 자동 설정), `params`
  - `_is_mergeable()`: self.adapter에서 직접 판단
  - 쿼리: `get_attr(node, idx)`, `get_attrs(nodes)`, `get_attrs_agg(node, agg_inner, agg_outer)`

- **SHAPCollector** (`_shap.py`): SHAP value 수집 및 분석
  - `explainer_cls`(default=shap.TreeExplainer), `data_filter`(DataFilter 인스턴스)
  - train/valid 각각 필터 적용 → SHAP 계산 → raw output 저장
  - 결과: `results[node][(idx, inner_idx)] = {'train', 'valid', 'train_index', 'valid_index', 'columns'}`
  - 분석: `get_feature_importance(node, idx)` → inner fold별 `pd.Series` 리스트
  - 분석: `get_feature_importance_agg(node, agg_inner='mean', agg_outer='mean')` → agg_inner=None이면 MultiIndex, agg_outer=None이면 DataFrame, 둘 다 설정이면 Series
  - SHAP 3D array(multiclass) 지원: `(n_samples, n_features, n_classes)` → class축 평균 후 처리

- **OutputCollector** (`_output.py`): output_train/output_valid 원본 저장
  - `output_var`, `include_target`
  - 파일 저장: `{path}/{node}/{idx}_{inner_idx}.pkl`
  - 쿼리: `get_output(node, idx, inner_idx)`, `get_outputs(node)`

## edges 구조
- dict 형태: `{key: [(node_name, var_spec), ...], ...}`
- key: 변수 집합 이름 (예: 'X', 'y', 'sample_weight')
- node_name: stage 노드명 (None=DataSource)
- var_spec: 변수 선택 (None=전체, str, list, callable, tuple)
- 같은 key의 데이터는 column 방향으로 concat
- 상위→하위 병합: 같은 key면 extend

## exist 파라미터 (set_grp, set_node, collect)
- `'diff'` (default, set_grp/set_node): 제공된 파라미터가 기존과 다를 때만 업데이트, 동일하면 skip
- `'skip'` (collect default): 이미 존재하면 무시하고 반환
- `'error'`: 이미 존재하면 ValueError
- `'replace'`: 기존 객체를 무조건 업데이트

### set_grp replace 동작
- `edges`, `params` 모두 `None→{}` 변환 후 모든 필드 직접 대입 (기존 값 유지 없음)

## Processor (`_node_processor.py`)
- **TransformProcessor**: `fit`, `fit_process`, `process`
- **PredictProcessor**: `fit`, `fit_process`, `process`
- `fit`/`fit_process`에서 y 데이터를 `squeeze()` 후 전달 (sklearn DataConversionWarning 억제)
- `get_feature_names_out` 반환값은 `list()` 로 변환하여 사용 (list/ndarray 호환)
- `data_dict` (Experimenter): `{key: ((train, train_v), valid), ...}` 형태
- `data_dict` (Trainer): `{key: (train, valid), ...}` 형태 (inner fold 없음)
- **X-less 지원**: `edges`에 `'X'`가 없고 `'y'`만 있는 경우(e.g. `LabelEncoder`) `'y'`를 primary input으로 사용
  - `fit`/`fit_process`: `'X'` 없으면 `'y'` 데이터를 squeeze하여 전달, `output_vars`를 `y_columns`로 설정
  - `process`: `X_`가 비어 있으면 입력 데이터를 squeeze 후 transform

## Adapter 인터페이스
- `get_params(params, logger)`: 모델 생성 파라미터
- `get_fit_params(data_dict, X, y, params, logger)`: fit 파라미터
- `result_objs`: `{name: (callable, mergeable_bool)}`
- `__eq__`: `type(self) is type(other) and self.__dict__ == other.__dict__` — diff 모드에서 adapter 비교에 사용
- `__hash__`: `id(self)` — set/dict 키로 사용 가능

## 보조 모듈
- **_data_wrapper.py**: DataWrapper (wrap/unwrap/squeeze/mean/mode/simple) — pandas/polars/cudf/numpy 통합
- **_describer.py**: desc_spec, desc_status, desc_pipeline, desc_node, desc_obj_vars (DataSource 기준)
- **_logger.py**: BaseLogger, DefaultLogger (start/update/end_progress, adhoc_progress)
- **col.py**: 컬럼 선택 유틸리티
- **_connector.py**: Connector (노드 매칭)
- **collector/**: Collector, MetricCollector, StackingCollector, ModelAttrCollector, SHAPCollector, OutputCollector
- **filter/**: DataFilter, RandomFilter(n/frac/random_state), IndexFilter(index)
- **adapter/**: sklearn, xgboost, lightgbm, catboost, keras
- **processor/**: CategoricalConverter, CategoricalPairCombiner, CatOOVFilter
  - polars 설치 시: PolarsLoader, ExprProcessor, PandasConverter 추가
  - `_dproc.py`: `get_type_df` (수치형만 f32/i32/i16/i8 판정), `get_type_pl`, `get_type_pd`, `merge_type_df`

## 저장 구조
```
{experimenter.path}/
  __exp.pkl                    # pipeline, node_obj_keys, collector_keys, 메타정보
  __collector/{name}/
    __config.pkl               # Collector 설정 + 데이터
    {node}.pkl                 # StackingCollector 노드별 데이터
    {node}/{idx}_{inner_idx}.pkl  # OutputCollector fold별 데이터
  {grp_path}/{node_name}/
    obj{idx}_{no}.pkl          # 빌드 결과 (StageObj/HeadObj)

{trainer.path}/
  __trainer.pkl                # name, splitter, selected_stages/heads, node_obj_keys, split_indices
  {grp_path}/{node_name}/
    obj{split_idx}.pkl         # 빌드 결과 (TrainStageObj/TrainHeadObj)

{inferencer_path}/
  __inferencer.pkl             # pipeline, selected_stages/heads, n_splits, node_objs, v (단일 파일)
```

## 패키지 정보
- PyPI 패키지명: `ml-labs`, Python 패키지: `mllabs/`
- `pyproject.toml`: setuptools 기반, Python >=3.10
- optional deps: `xgboost`, `lightgbm`, `catboost`, `shap`, `polars`, `all`, `dev`
- 릴리즈: `v*` 태그 push → GitHub Actions (`publish.yml`) → 테스트(3.10/3.11/3.12) → build → PyPI 자동 배포 (OIDC)

## 향후 방향
- Experimenter에서 도출한 Pipeline으로 Train/Inference 파이프라인 구성 → test 데이터 예측
