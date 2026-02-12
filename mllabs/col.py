"""
Column selection helper functions for resolve_columns
"""
import re

def ohe_drop_first(columns, processor):
    # 각 원래 변수에 대해 첫 번째 컬럼을 만났는지 추적
    org_X = processor.X_
    first_seen = {var: False for var in org_X}

    mask = []
    for col in columns:
        # '__' 뒤의 부분 추출
        if '__' not in col:
            mask.append(False)
            continue

        suffix = col.split('__', 1)[1]

        # 어떤 org_X 변수에 속하는지 확인
        matched = False
        for org_var in org_X:
            # suffix가 org_var로 시작하는지 확인 (예: color_red는 color로 시작)
            if suffix.startswith(f"{org_var}_"):
                matched = True
                if not first_seen[org_var]:
                    # 첫 번째 컬럼은 제외 (False)
                    mask.append(False)
                    first_seen[org_var] = True
                else:
                    # 나머지는 포함 (True)
                    mask.append(True)
                break

        # org_X의 어떤 변수에도 속하지 않으면 제외
        if not matched:
            mask.append(False)

    return mask

def _polynomial_feature_names(input_features, degree=2, interaction_only=False, include_bias=True):
    from itertools import combinations, combinations_with_replacement
    n_features = len(input_features)
    feature_names = []
    if include_bias:
        feature_names.append("1")
    comb_func = combinations if interaction_only else combinations_with_replacement
    for d in range(1, degree + 1):
        for comb in comb_func(range(n_features), d):
            counts = {}
            for idx in comb:
                counts[idx] = counts.get(idx, 0) + 1
            terms = []
            for idx, power in counts.items():
                name = input_features[idx]
                if power > 1:
                    name = f"{name}^{power}"
                terms.append(name)
            feature_names.append(" ".join(terms))
    return feature_names

def subset_poly(columns, vars, processor=None):
    obj = processor.obj
    degree = obj.degree if hasattr(obj, 'degree') else 2
    interaction_only = obj.interaction_only if hasattr(obj, 'interaction_only') else False
    include_bias = obj.include_bias if hasattr(obj, 'include_bias') else True

    org_X = processor.X_
    vars_ = list()
    for x in org_X:
        for i in vars:
            if i == x or re.match(i, x):
                vars_.append(x)
                break

    subset_names = set(_polynomial_feature_names(
        vars_, degree=degree, interaction_only=interaction_only, include_bias=include_bias
    ))

    node_name = processor.name
    prefix = f"{node_name}__"

    mask = []
    for col in columns:
        if col.startswith(prefix):
            suffix = col[len(prefix):]
            mask.append(suffix in subset_names)
        else:
            mask.append(False)
    return mask

def get_origin_var(columns, org_X):
    l = list()
    for col in columns:
        suffix = col.split('__', 1)[-1]
        org = "Unknown"
        for org_var in org_X:
            if suffix.startswith(f'{org_var}_') or suffix == org_var:
                org = org_var
                break
        l.append(org)
    return l
