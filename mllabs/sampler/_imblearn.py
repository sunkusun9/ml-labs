from ._base import Sampler


class ImbLearnSampler(Sampler):
    def __init__(self, sampler):
        self.sampler = sampler

    def sample(self, fit_params):
        X = fit_params.get('X')
        y = fit_params.get('y')
        X_res, y_res = self.sampler.fit_resample(X, y)
        result = dict(fit_params)
        result['X'] = X_res
        if y is not None:
            result['y'] = y_res
        return result
