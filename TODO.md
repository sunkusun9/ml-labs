# TODO (feat/issue-107-collector-refactor)

## 완료
- [x] `BuildWorker` 클래스 구현 (`_expobj.py`) — inner fold 단위 병렬 처리, 파일 기반 결과
- [x] `_build_iter` generator로 전환 — `(obj, result, info, coll_dict)` yield
- [x] `_build_iter_output` 제거, `_dispatch_iter_output`으로 대체
- [x] `Collector` 인터페이스 재설계 (`_base.py`) — `collect`, `push`, `end_idx`
- [x] `DataFlow` / `TrainDataFlow` 설계 및 구현 (`_flow.py`)
- [x] `DataSourceProvider` abstract interface 정의

## 진행 예정

### Collector 체계 정리
- [ ] Collector 별 `push` / `end_idx` 구현 가이드 확정 후 적용
  - `MetricCollector`: inner fold 결과 조각화 심함 → outer fold 단위 병합 전략 필요
  - `StackingCollector`, `ModelAttrCollector`, `SHAPCollector`, `OutputCollector` 각각 결정
- [ ] `BuildWorker`가 남긴 per-inner-fold `_collect_{idx}_{no}.pkl` 파일을 outer fold 종료 후 collector로 통합하는 루틴 구현 (collect 체계 정리 후)

### DataFlow / TrainDataFlow 통합
- [ ] `DataSourceProvider` 구체 구현
  - `ExperimenterDataSource`: outer×inner fold 인덱스 기반
  - `TrainerDataSource`: split 인덱스 기반
- [ ] `Experimenter`에 `train_flows[outer][inner]` 2D 배열 통합 (기존 StageObj 로직 대체)
- [ ] `Trainer`에 `TrainDataFlow` 통합

### 병렬 실행
- [ ] `Experimenter.build` / `exp`에서 `BuildWorker`를 활용한 실제 병렬 실행 연결
