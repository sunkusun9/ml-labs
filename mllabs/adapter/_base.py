"""
Base adapter class for ML model frameworks
"""

from abc import ABC, abstractmethod

GPU_NO = 'no'
GPU_POSSIBLE = 'possible'
GPU_YES = 'yes'


class ModelAdapter(ABC):
    """Abstract base class for ML framework adapters.

    Adapters translate ml-labs' unified data format into the framework-specific
    ``fit()`` parameters of each model. Registered by model class name via
    :func:`~mllabs.adapter.register_adapter`.

    Class Attributes:
        result_objs (dict): ``{key: (callable, mergeable_bool)}`` mapping of
            extractable model attributes for use with
            :class:`~mllabs.collector.ModelAttrCollector`.
    """

    result_objs = {}
    """Abstract base class for model adapters

    각 머신러닝 프레임워크별로 eval_set 처리 방식이 다르므로,
    이를 통일된 인터페이스로 추상화합니다.
    """

    def __init__(self, eval_mode='both', verbose=0.1):
        """Adapter 초기화

        Args:
            eval_mode (str): Evaluation mode
                - 'none' or None: eval_set 없이
                - 'valid': validation set만 전달
                - 'both': train + validation set 전달
            verbose: Verbose 설정
                - 0: 출력 안함
                - 0 < verbose < 1: 전체 진행률을 % 단위로 표시 (주기: verbose * 100%)
                  예: 0.1이면 10%마다, 0.05면 5%마다
                - verbose >= 1: iteration 단위로 표시 (매 verbose번째 iteration)
                  예: 1이면 매 iteration마다, 10이면 매 10 iteration마다
        """
        self.eval_mode = eval_mode
        self.verbose = verbose

    def get_fit_params(self, data_dict, params=None, logger=None):
        """모델의 fit()에 전달할 파라미터를 구성

        Args:
            data_dict: {key: (train, train_v), ...} 형태의 데이터 딕셔너리
            params (dict): Processor에서 전달된 추가 파라미터 (Optional, default=None)
            logger: Logger 인스턴스

        Returns:
            dict: fit()에 unpacking으로 전달할 파라미터
                  예: model.fit(**fit_params)
        """
        from .._data_wrapper import unwrap
        fit_params = {}
        if 'X' in data_dict:
            fit_params['X'] = unwrap(data_dict['X'][0])
        if 'y' in data_dict:
            fit_params['y'] = unwrap(data_dict['y'][0].squeeze())
        return fit_params

    def get_process_data(self, data):
        from .._data_wrapper import unwrap
        return unwrap(data)

    def get_gpu_usage(self, params):
        """Returns whether the current params will use GPU.

        Returns:
            'no'      — GPU not used
            'possible' — GPU may be used (framework auto-selects based on hardware)
            'yes'     — GPU will definitely be used
        """
        gpu = (params or {}).get('gpu', 'auto')
        if gpu is None:
            return GPU_NO
        if gpu == 'auto':
            return GPU_NO
        return GPU_YES

    def get_params(self, params, logger=None):
        """모델 생성자에 전달할 파라미터를 조정

        Args:
            params (dict): 원본 파라미터

        Returns:
            dict: 조정된 파라미터. gpu 키는 제거됨
        """
        if params is None:
            return params
        params = params.copy()
        params.pop('gpu', None)
        return params

    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __hash__(self):
        return id(self)
