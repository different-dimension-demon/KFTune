from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import Matern
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from scipy.stats import norm
import numpy as np


class BayesianOptimizer:
    def __init__(self, surrogate_type="GP", acquisition_type="PI"):
        self.surrogate_type = surrogate_type.upper()
        self.acquisition_type = acquisition_type.upper()
        self.rng = np.random.RandomState(42)

    def _fit_surrogate(self, X, y):
        if self.surrogate_type == "GP":
            kernel = Matern(nu=2.5)
            model = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        elif self.surrogate_type == "RF":
            model = RandomForestRegressor(n_estimators=100)
        elif self.surrogate_type == "GBRT":
            model = GradientBoostingRegressor(n_estimators=100)
        elif self.surrogate_type == "KRR":
            model = KernelRidge(kernel='rbf')
        elif self.surrogate_type == "SVR":
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        else:
            raise ValueError(f"Unsupported surrogate model: {self.surrogate_type}")

        model.fit(X, y)
        return model

    def _acquisition(self, X, model, X_sample, y_sample):
        if self.acquisition_type == "EI":
            return self._expected_improvement(X, model, y_sample)
        elif self.acquisition_type == "PI":
            return self._probability_improvement(X, model, y_sample)
        elif self.acquisition_type == "UCB":
            return self._upper_confidence_bound(X, model)
        elif self.acquisition_type == "TS":  
            return self._thompson_sampling(X, model)
        else:
            raise ValueError(f"Unsupported acquisition function: {self.acquisition_type}")

    def _expected_improvement(self, X, model, y_sample, xi=0.01):
        mu, sigma = model.predict(X, return_std=True)
        sigma = sigma.ravel()
        mu_sample_opt = np.min(y_sample)

        with np.errstate(divide='warn'):
            imp = mu_sample_opt - mu - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def _probability_improvement(self, X, model, y_sample, xi=0.01):
        mu, sigma = model.predict(X, return_std=True)
        mu_sample_opt = np.min(y_sample)
        Z = (mu_sample_opt - mu - xi) / sigma
        pi = norm.cdf(Z)
        return pi

    def _upper_confidence_bound(self, X, model, kappa=2.576):
        mu, sigma = model.predict(X, return_std=True)
        return mu - kappa * sigma
    
    def _thompson_sampling(self, X, model):
        if not hasattr(model, "sample_y"):
            return model.predict(X) + self.rng.normal(0, 1e-6, size=X.shape[0])
        sample = model.sample_y(X, n_samples=1).ravel()
        return sample

    def suggest(self, X_sample, y_sample, n_cont, n_total, n_candidates=100):
        model = self._fit_surrogate(X_sample, y_sample)

        candidates = np.zeros((n_candidates, n_total))
        for i in range(n_total):
            candidates[:, i] = self.rng.uniform(0.0, 1.0, size=n_candidates)

        acq = self._acquisition(candidates, model, X_sample, y_sample)
        next_x = candidates[np.argmax(acq)]
        return next_x
