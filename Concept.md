# modeler Concept

## Node 상태 모델

### Node 4-State
```
init ─→ built ─→ finalized
  │
  └─→ error ─→ (reset) ─→ init
```

| 상태 | NodeObj | Disk | Memory | 설명 |
|------|---------|------|--------|------|
| **init** | 없음 | - | - | Pipeline에 정의만 된 상태 |
| **built** | 생성됨 | O | Stage: O / Head: X | 빌드 완료, 결과 추출 가능 |
| **finalized** | 상태만 잔존 | X | X | 결과 추출 완료, 리소스 해제 |
| **error** | 에러 정보 보유 | - | - | 빌드/실험 중 에러 발생, 내역 보존 |

- **finalize는 Head 전용**, Stage는 차단
- Stage Node: 하위 노드에 지속적으로 데이터를 공급해야 하므로 메모리 유지
- Head Node: 결과 추출 시에만 사용하므로 Disk에만 저장
- 역방향 전이 없음 (다시 하려면 reset → rebuild)

### error 상태
- build/exp 중 에러 발생 시 Exception을 raise하지 않고 error 상태로 전환
- 에러 내용을 저장하여 사후 디버깅 가능
- 나머지 노드는 실험 계속 진행
- 상위 Stage가 error면 하위 노드도 입력 데이터를 못 받아 자연스럽게 error — 별도 전파 로직 불필요
- error → reset → init으로 복구 후 재실행

### Experiment 2-State
```
open → closed
```

| 상태 | Stage 객체 | Head 객체 | Collector 데이터 | 설명 |
|------|-----------|-----------|-----------------|------|
| **open** | 유지 | 유지/finalized | 유지 | 실험 진행 중 |
| **closed** | 제거 | 제거 | 유지 | 실험 종료, 수집 데이터만 잔존 |

- `close_exp`로 전환: Stage 객체까지 일괄 정리
- Collector 데이터는 독립 저장이라 closed 후에도 조회 가능 (메트릭, 스태킹 등)

## 리소스 관리 설계

### Cache 계층
- DataCache (LRU, 용량 기반): built 상태 내에서 투명하게 동작
- Cache miss 시 disk의 객체로 재생성 → 별도 상태 불필요

### 리소스 생명주기
```
build → 메모리/디스크 확보
  → 실험 수행 (cache 활용)
  → Head finalize → Head 리소스 해제
  → close_exp → Stage 리소스까지 해제
  → Collector 데이터만 잔존
```

## Pipeline 구조

### Node 역할
- **DataSource** (None): 원본 데이터 제공, Pipeline에 명시적 노드 없음
- **Stage**: 전처리/변환 (TransformProcessor) — 하위 노드에 데이터 공급
- **Head**: 모델링/예측 (PredictProcessor) — 최종 결과 생산

### edges 구조
```python
{key: [(stage_node_name, var_spec), ...]}
```
- key: 변수 집합 이름 ('X', 'y', 'sample_weight' 등)
- stage_node_name: None이면 DataSource
- var_spec: None(전체), str, list, callable, tuple
- 같은 key 내 데이터는 column concat
- 상위 그룹 → 하위 그룹: 같은 key면 extend

### Group 계층
- PipelineGroup: 노드의 논리적 그룹 (속성 상속)
- 상위 그룹의 processor, params, edges, method, adapter를 하위로 병합
- 노드 개별 속성이 그룹 속성을 override

## Collector 체계

### 설계 원칙
- Connector 기반 자동 매칭: 어떤 노드에 어떤 Collector를 붙일지 선언적으로 결정
- 독립 저장: Experimenter 상태와 분리, close_exp 후에도 데이터 유지
- 라이프사이클: `_start` → `_collect` (per inner fold) → `_end_idx` (per outer fold) → `_end`

### Collector 종류
| Collector | 수집 대상 | 저장 방식 | 용도 |
|-----------|----------|----------|------|
| MetricCollector | scalar metric | 메모리 + pkl | 성능 비교 |
| StackingCollector | OOF 예측값 (집계) | 파일 or 메모리 | 2nd level 모델 |
| ModelAttrCollector | 모델 속성 (feature importance 등) | 메모리 + pkl | 모델 해석 |
| SHAPCollector | SHAP values | 메모리 + pkl | 피처 기여도 분석 |
| OutputCollector | output_train/output_valid 원본 | 파일 (per fold) | 사후 분석 (confusion matrix 등) |

## Adapter 패턴
- 프레임워크별 차이를 흡수 (sklearn, LightGBM, XGBoost, CatBoost, Keras)
- `get_params`: 모델 생성 파라미터
- `get_fit_params`: fit 파라미터 (eval_set, callbacks 등)
- `result_objs`: 모델 속성 추출 정의 `{name: (callable, mergeable)}`

## 향후 방향
- Experimenter에서 도출한 Pipeline으로 Train/Inference 파이프라인 구성
  → test 데이터 예측이 자연스럽게 해결
- 패키지화 (이름 미정)
