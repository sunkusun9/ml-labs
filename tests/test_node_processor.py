import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from mllabs._data_wrapper import PandasWrapper
from mllabs._node_processor import TransformProcessor


def make_wrapper(df_or_array, columns=None, index=None):
    if isinstance(df_or_array, pd.DataFrame):
        return PandasWrapper(df_or_array)
    arr = np.array(df_or_array)
    df = pd.DataFrame(arr, columns=columns, index=index)
    return PandasWrapper(df)


def make_data_dict(train_X=None, train_v_X=None,
                   train_y=None, train_v_y=None):
    d = {}
    if train_X is not None:
        d['X'] = (make_wrapper(train_X), make_wrapper(train_v_X))
    if train_y is not None:
        d['y'] = (make_wrapper(train_y), make_wrapper(train_v_y))
    return d


@pytest.fixture
def x_data():
    train = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [10.0, 20.0, 30.0, 40.0]})
    train_v = pd.DataFrame({'a': [1.5, 2.5], 'b': [15.0, 25.0]})
    valid = pd.DataFrame({'a': [5.0, 6.0], 'b': [50.0, 60.0]})
    return train, train_v, make_wrapper(valid)


@pytest.fixture
def y_data():
    train = pd.DataFrame({'label': ['Absence', 'Presence', 'Absence', 'Presence']})
    train_v = pd.DataFrame({'label': ['Absence', 'Presence']})
    valid = pd.DataFrame({'label': ['Presence', 'Absence']})
    return train, train_v, make_wrapper(valid)


class TestTransformProcessorWithX:
    def test_fit_sets_X_(self, x_data):
        train, train_v, valid = x_data
        data_dict = make_data_dict(train, train_v)
        proc = TransformProcessor('std', StandardScaler)
        proc.fit(data_dict)
        assert proc.X_ == ['a', 'b']

    def test_fit_creates_obj(self, x_data):
        train, train_v, valid = x_data
        data_dict = make_data_dict(train, train_v)
        proc = TransformProcessor('std', StandardScaler)
        proc.fit(data_dict)
        assert isinstance(proc.obj, StandardScaler)

    def test_fit_process_returns_wrapper(self, x_data):
        train, train_v, valid = x_data
        data_dict = make_data_dict(train, train_v)
        proc = TransformProcessor('std', StandardScaler)
        result = proc.fit_process(data_dict)
        assert isinstance(result, PandasWrapper)
        assert result.get_columns() == ['std__a', 'std__b']
        assert len(result.to_array()) == 4

    def test_process_transforms_valid(self, x_data):
        train, train_v, valid = x_data
        data_dict = make_data_dict(train, train_v)
        proc = TransformProcessor('std', StandardScaler)
        proc.fit_process(data_dict)
        result = proc.process(valid)
        assert isinstance(result, PandasWrapper)
        assert result.get_columns() == ['std__a', 'std__b']

    def test_fit_process_output_vars_from_get_feature_names_out(self, x_data):
        train, train_v, valid = x_data
        data_dict = make_data_dict(train, train_v)
        proc = TransformProcessor('ohe', OneHotEncoder, params={'sparse_output': False})
        proc.fit_process(data_dict)
        assert proc.output_vars is not None
        assert all(col.startswith('ohe__') for col in proc.output_vars)


class TestTransformProcessorNoX:
    def test_fit_without_x_sets_empty_X_(self, y_data):
        train, train_v, valid = y_data
        data_dict = make_data_dict(train_y=train, train_v_y=train_v)
        proc = TransformProcessor('le', LabelEncoder)
        proc.fit(data_dict)
        assert proc.X_ == []

    def test_fit_without_x_creates_obj(self, y_data):
        train, train_v, valid = y_data
        data_dict = make_data_dict(train_y=train, train_v_y=train_v)
        proc = TransformProcessor('le', LabelEncoder)
        proc.fit(data_dict)
        assert isinstance(proc.obj, LabelEncoder)
        assert hasattr(proc.obj, 'classes_')

    def test_fit_without_x_encodes_correctly(self, y_data):
        train, train_v, valid = y_data
        data_dict = make_data_dict(train_y=train, train_v_y=train_v)
        proc = TransformProcessor('le', LabelEncoder)
        proc.fit(data_dict)
        classes = list(proc.obj.classes_)
        assert 'Absence' in classes
        assert 'Presence' in classes

    def test_fit_without_x_sets_output_vars_to_y_columns(self, y_data):
        train, train_v, valid = y_data
        data_dict = make_data_dict(train_y=train, train_v_y=train_v)
        proc = TransformProcessor('le', LabelEncoder)
        proc.fit(data_dict)
        assert proc.output_vars == ['label']

    def test_fit_process_without_x_returns_encoded(self, y_data):
        train, train_v, valid = y_data
        data_dict = make_data_dict(train_y=train, train_v_y=train_v)
        proc = TransformProcessor('le', LabelEncoder)
        result = proc.fit_process(data_dict)
        assert isinstance(result, PandasWrapper)
        assert result.get_columns() == ['label']
        arr = result.to_array().ravel()
        assert set(arr).issubset({0, 1})

    def test_fit_process_without_x_presence_is_1(self, y_data):
        train, train_v, valid = y_data
        data_dict = make_data_dict(train_y=train, train_v_y=train_v)
        proc = TransformProcessor('le', LabelEncoder)
        proc.fit_process(data_dict)
        presence_encoded = proc.obj.transform(['Presence'])[0]
        assert presence_encoded == 1

    def test_process_without_x_transforms_valid(self, y_data):
        train, train_v, valid = y_data
        data_dict = make_data_dict(train_y=train, train_v_y=train_v)
        proc = TransformProcessor('le', LabelEncoder)
        proc.fit_process(data_dict)
        result = proc.process(valid)
        assert isinstance(result, PandasWrapper)
        arr = result.to_array().ravel()
        assert set(arr).issubset({0, 1})

    def test_process_without_x_correct_encoding(self, y_data):
        train, train_v, valid = y_data
        data_dict = make_data_dict(train_y=train, train_v_y=train_v)
        proc = TransformProcessor('le', LabelEncoder)
        proc.fit_process(data_dict)
        # valid has ['Presence', 'Absence']
        result = proc.process(valid)
        arr = result.to_array().ravel()
        absence_val = proc.obj.transform(['Absence'])[0]
        presence_val = proc.obj.transform(['Presence'])[0]
        assert arr[0] == presence_val
        assert arr[1] == absence_val
