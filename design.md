# NN Estimator Design — Issue #82

## Goals

- sklearn-compatible `NNClassifier` / `NNRegressor`
- 모델을 **Head / Body / Tail** 세 요소로 분리, 자유롭게 조합
- categorical columns → entity embedding (per-column 크기 자동/수동)
- pandas/polars `category` dtype 자동 감지
- early stopping + `evals_result_` (XGB/LGB 패턴 통일)
- ml-labs `NNAdapter` 통해 Pipeline Head 노드로 사용 가능

---

## 빌드 순서

```
1. mllabs/nn/ 패키지 + TF 구현 완성
2. 공통 / TF 전용 요소 분리 식별
3. _NNBackend 추상 레이어 필요 여부 결정 (PyTorch 추가 시점에)
4. NNAdapter 구현
5. (나중에) 직렬화 전략 — Adapter 공통 터치포인트로 처리
```

---

## 패키지 위치

```
mllabs/
  nn/
    __init__.py        # NNClassifier, NNRegressor, SimpleConcatHead, DenseBody, LogitTail, RegressionTail
    _estimator.py      # NNClassifier, NNRegressor (_NNBase)
    _encoder.py        # _CatEncoder
    _head.py           # NNHead (ABC), EmbeddingHead
    _body.py           # NNBody (ABC), DenseBody
    _tail.py           # NNTail (ABC), LogitTail, RegressionTail
  adapter/
    _nn.py             # NNAdapter
```

---

## 아키텍처 개요

```
X (DataFrame / ndarray)
  │
  ▼  _CatEncoder.transform()
  ├─ cat_arrays  [n_cat × (n,) int32]
  └─ cont_array  (n, n_cont) float32
          │
          ▼  Head.build()
    단일 Tensor  (n, emb_concat_dim + n_cont)
          │
          ▼  Body.build()
    단일 Tensor  (n, hidden_dim)
          │
          ▼  Tail.build()
      Output  (n, n_output)
```

Head / Body / Tail은 모두 **Keras functional API 텐서를 반환하는 builder 객체**.
TF 의존 없이 config만 보유하다가 `build()` 호출 시 Keras 그래프 구성.
→ sklearn `BaseEstimator` 서브클래스로 만들어 `get_params()`/`set_params()` 지원.

---

## Head

입력(cat embeddings + cont)을 **하나의 Tensor**로 변환.
Head의 출력 shape이 Body와의 **호환 계약**이 된다.

```python
class NNHead(BaseEstimator, ABC):
    @abstractmethod
    def build(self, cat_inputs, cont_input, cat_specs):
        # cat_inputs  : list[tf.Tensor]  각 cat col의 Keras Input (shape=(1,), dtype=int32)
        # cont_input  : tf.Tensor | None  (shape=(n_cont,), dtype=float32)
        # cat_specs   : list[(col_name, n_unique, emb_dim)]
        # return      : tf.Tensor
        ...
```

### SimpleConcatHead (기본)

각 cat → 크기가 다른 embedding → Flatten, cont → 그대로, 전부 **Concatenate**.
출력: `(batch, sum(emb_dims) + n_cont)` — **2D**.

```python
class SimpleConcatHead(NNHead):
    def __init__(self, dropout=0.0):
        self.dropout = dropout   # embedding dropout (선택)

    def build(self, cat_inputs, cont_input, cat_specs):
        parts = []
        for inp, (_, n_unique, emb_dim) in zip(cat_inputs, cat_specs):
            emb = Embedding(n_unique + 1, emb_dim)(inp)   # (batch, 1, emb_dim)
            emb = Flatten()(emb)                           # (batch, emb_dim)
            if self.dropout > 0:
                emb = Dropout(self.dropout)(emb)
            parts.append(emb)
        if cont_input is not None:
            parts.append(cont_input)
        return Concatenate()(parts) if len(parts) > 1 else parts[0]
```

→ `DenseBody`와 pair.

---

### FTTokenizerHead (확장 예정)

FT-Transformer의 Feature Tokenizer 구조.
cat과 cont 모두 **동일한 d_token 크기**의 벡터로 변환 후 stack.
출력: `(batch, n_features, d_token)` — **3D** (feature 차원 추가).

```
categorical col i  → Embedding(n_unique_i + 1, d_token)       → (batch, d_token)
continuous  col j  → x_j * W_j + b_j  (W_j ∈ R^d_token)      → (batch, d_token)
                                                                     ↓ stack
                                                       (batch, n_features, d_token)
```

→ `TransformerBody`와 pair. Body가 3D 입력을 받아 Attention 후 CLS 토큰 → 2D 반환.

> Head 출력 shape summary:
> | Head | 출력 shape | 대응 Body |
> |------|-----------|-----------|
> | `SimpleConcatHead` | 2D `(batch, D)` | `DenseBody` |
> | `FTTokenizerHead` | 3D `(batch, n_feat, d_token)` | `TransformerBody` |

---

## Body

Head 출력 → 은닉 표현. **TF Layer / Model 어떤 것이든 자유롭게 구성**.

```python
class NNBody(BaseEstimator, ABC):
    @abstractmethod
    def build(self, x):
        # x      : tf.Tensor — Head 출력
        # return : tf.Tensor — 은닉 표현
        ...
```

### DenseBody (기본)

```python
class DenseBody(NNBody):
    def __init__(self, layers=(256, 128), dropout=0.3, activation="relu", batch_norm=False):
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm

    def build(self, x):
        for units in self.layers:
            x = Dense(units)(x)
            if self.batch_norm:
                x = BatchNormalization()(x)
            x = Activation(self.activation)(x)
            if self.dropout > 0:
                x = Dropout(self.dropout)(x)
        return x
```

> 확장 예: `ResidualBody`, `LSTMBody`, 또는 사용자가 직접 Keras Layer를 감싼 커스텀 Body.

---

## Tail

Body 출력 → 최종 출력. 태스크별로 교체.

```python
class NNTail(BaseEstimator, ABC):
    @abstractmethod
    def build(self, x, n_output):
        # x        : tf.Tensor — Body 출력
        # n_output : int — 출력 차원 (NNClassifier/NNRegressor가 주입)
        # return   : tf.Tensor
        ...

    @abstractmethod
    def loss(self):
        # Keras loss 문자열 또는 객체 반환
        ...

    @abstractmethod
    def compile_metrics(self):
        # Keras metrics 리스트 반환
        ...
```

### 기본 구현

```python
class LogitTail(NNTail):
    """다중 클래스 분류"""
    def build(self, x, n_output):
        return Dense(n_output, activation="softmax")(x)
    def loss(self): return "sparse_categorical_crossentropy"
    def compile_metrics(self): return ["accuracy"]

class BinaryLogitTail(NNTail):
    """이진 분류"""
    def build(self, x, n_output):
        return Dense(1, activation="sigmoid")(x)
    def loss(self): return "binary_crossentropy"
    def compile_metrics(self): return ["accuracy"]

class RegressionTail(NNTail):
    """회귀"""
    def build(self, x, n_output):
        return Dense(n_output)(x)
    def loss(self): return "mse"
    def compile_metrics(self): return ["mae"]
```

> 확장 예: `NegativeSamplingTail`, `RankingTail`, `PoissonTail` 등 동일 인터페이스로 추가.

---

## NNClassifier / NNRegressor

```python
class NNClassifier(_NNBase, ClassifierMixin):
    def __init__(
        self,
        cat_cols=None,
        embedding_dims=None,
        head=None,             # None → EmbeddingHead()
        body=None,             # None → DenseBody()
        tail=None,             # None → LogitTail() or BinaryLogitTail() (클래스 수로 자동)
        epochs=100,
        batch_size=1024,
        learning_rate=1e-3,
        early_stopping_patience=10,
        validation_fraction=0.1,
        random_state=None,
    )

class NNRegressor(_NNBase, RegressorMixin):
    def __init__(
        self,
        ...
        tail=None,             # None → RegressionTail()
        ...
    )
```

**_NNBase 책임**:
- `_CatEncoder` 관리 (fit/transform)
- `embedding_dims_` 확정 (auto + override)
- Keras Input 텐서 생성
- Head/Body/Tail `.build()` 호출 → `tf.keras.Model` 조립
- `fit()`, `predict()` 구현
- `evals_result_`, `best_epoch_` 저장

---

## fit 시그니처 & evals_result_

```python
def fit(self, X, y, eval_set=None):
    # eval_set=[(X_val, y_val)]  — XGB/LGB 패턴
    # None이면 validation_fraction으로 내부 split
```

fit 완료 후:
```python
evals_result_ = {
    "train": {"loss": [...], "accuracy": [...]},
    "valid": {"loss": [...], "accuracy": [...]},   # eval_set 있을 때만
}
best_epoch_: int
```

---

## 전체 Keras 모델 조립 (_NNBase._build_model)

```python
def _build_model(self):
    # 1. Inputs
    cat_inputs = [Input(shape=(1,), dtype='int32', name=f'cat_{col}')
                  for col in self.encoder_.cat_cols_]
    n_cont = len(self.encoder_.cont_cols_)
    cont_input = Input(shape=(n_cont,), name='cont') if n_cont > 0 else None

    # 2. Head
    cat_specs = [(col, self.encoder_.cardinalities_[col], self.embedding_dims_[col])
                 for col in self.encoder_.cat_cols_]
    x = self.head_.build(cat_inputs, cont_input, cat_specs)

    # 3. Body
    x = self.body_.build(x)

    # 4. Tail
    output = self.tail_.build(x, self._n_output())

    # 5. Compile
    inputs = cat_inputs + ([cont_input] if cont_input else [])
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(self.learning_rate),
                  loss=self.tail_.loss(),
                  metrics=self.tail_.compile_metrics())
    return model
```

---

## _CatEncoder

`fit(X)` 시점:
1. `cat_cols` 확정 (None → category dtype 자동, numpy+None → ValueError)
2. 컬럼별 카테고리 목록 수집 → `encoders_: dict {col: [cat0, ...]}`
3. `cardinalities_: dict {col: n_unique}`
4. `cont_cols_`: cat_cols 제외 나머지

`transform(X)` 반환:
```python
(
    cat_arrays,   # list[np.ndarray (n,) int32]
    cont_array,   # np.ndarray (n, n_cont) float32
)
```

OOV: unseen → 0 (최소 방어). 본격 처리는 upstream `CatOOVFilter` 위임.

---

## NNAdapter

```python
class NNAdapter(ModelAdapter):
    def get_fit_params(self, data_dict, params=None, logger=None):
        train_v_X = unwrap(data_dict['X'][0][1])
        train_v_y = unwrap(data_dict['y'][0][1]) if 'y' in data_dict else None
        if self.eval_mode != 'none' and train_v_X is not None:
            return {'eval_set': [(train_v_X, train_v_y)]}
        return {}

NNAdapter.result_objs = {
    'evals_result': (_get_evals_result, True),
}
```

---

## 미결 / 나중에

| 항목 | 상태 |
|------|------|
| `_NNBackend` 추상화 | PyTorch 추가 시점에 결정 |
| 직렬화 | 구현 완료 후 Adapter 공통 처리 |
| OOV 전용 처리 | upstream `CatOOVFilter` 위임, 우선순위 낮음 |
| `BinaryLogitTail` 자동 선택 기준 | n_classes==2 → Binary, else → Logit |
| multioutput regressor | 일단 단일 타겟만 |
| `get_feature_names_out` | 나중에 |
| PyTorch 백엔드 | v0.7.0 이후 |
| `NegativeSamplingTail`, `RankingTail` 등 | 별도 이슈 |
| `FTTokenizerHead` + `TransformerBody` | 별도 이슈 |
| embedding dropout, regularization 등 Head 확장 | 별도 이슈 |
