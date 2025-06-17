from typing import Dict, List
import numpy as np
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from skopt.space import Real
from skopt.sampler import Lhs

class ConfigEncoder:
    def __init__(self, config_space: ConfigurationSpace, tau: float = 1.0):
        self.config_space = config_space 
        self.continuous_params = []
        self.discrete_params = {}
        self.param_bounds = {}
        self.tau = tau
        self.best_config = None
        self.selected_knobs = []

        for hp in config_space.get_hyperparameters():
            if isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
                self.continuous_params.append(hp.name)
                self.param_bounds[hp.name] = (hp.lower, hp.upper)
            elif isinstance(hp, CategoricalHyperparameter):
                self.discrete_params[hp.name] = hp.choices

        self.discrete_value_to_index = {
            param: {v: i for i, v in enumerate(choices)}
            for param, choices in self.discrete_params.items()
        }
        self.discrete_index_to_value = {
            param: choices
            for param, choices in self.discrete_params.items()
        }

        self.rng = np.random.RandomState(42)

    def sample_gumbel(self, shape, eps=1e-20):
        U = self.rng.rand(*shape)
        return -np.log(-np.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, tau):
        gumbel_noise = self.sample_gumbel(logits.shape)
        y = logits + gumbel_noise
        return np.exp(y / tau) / np.sum(np.exp(y / tau), axis=-1, keepdims=True)

    def encode(self, config: Dict) -> np.ndarray:
        encoded = []
        for param in self.continuous_params:
            value = config[param]
            lower, upper = self.param_bounds[param]
            normalized = (value - lower) / (upper - lower)
            encoded.append(normalized)
        for param in self.discrete_params:
            choices = self.discrete_index_to_value[param]
            idx = self.discrete_value_to_index[param][config[param]]
            logits = np.zeros(len(choices))
            logits[idx] = 1.0  # one-hot init
            softmax_sample = self.gumbel_softmax_sample(logits, self.tau)
            encoded.extend(softmax_sample)
        return np.array(encoded, dtype=np.float64)

    def decode(self, encoded: np.ndarray) -> Dict:
        decoded = {}
        index = 0
        for param in self.continuous_params:
            lower, upper = self.param_bounds[param]
            normalized = encoded[index]
            value = lower + normalized * (upper - lower)
            decoded[param] = np.clip(value, lower, upper)
            index += 1

        for param in self.discrete_params:
            choices = self.discrete_index_to_value[param]
            num_choices = len(choices)
            probs = encoded[index: index + num_choices]
            probs = probs / np.sum(probs)  # Ensure sum to 1 (probability distribution)
        
            # Perform probabilistic sampling
            idx = self.rng.choice(np.arange(num_choices), p=probs)
            decoded[param] = choices[idx]
            index += num_choices

        return decoded
    
    def decode_index(self, indices, encoded) -> Dict:
        knobs = {}
        for index in indices:
            if index <= len(self.continuous_params)-1:
                param = self.continuous_params[index]
                lower, upper = self.param_bounds[param]
                normalized = encoded[index]
                value = lower + normalized * (upper - lower)
                knobs[param] = np.clip(value, lower, upper)
            else:
                end = len(self.continuous_params)-1
                for param in self.discrete_params:
                    choices = self.discrete_index_to_value[param]
                    num_choices = len(choices)
                    end = end + num_choices
                    if index <= end:
                        probs = encoded[end-num_choices: end]
                        probs = probs / np.sum(probs)  
                        idx = self.rng.choice(np.arange(num_choices), p=probs)
                        knobs[param] = choices[idx]
                        break
        return knobs

    def get_encoded_dim(self):
        return len(self.continuous_params) + sum(len(choices) for choices in self.discrete_index_to_value.values())

    def get_continuous_dim(self):
        return len(self.continuous_params)

    def sample_random_configs(self, n: int = 1) -> List[Dict]:
        configs = []
        dimensions = [Real(*self.param_bounds[param]) for param in self.continuous_params]
        lhs = Lhs(lhs_type="classic", criterion=None)
        samples = lhs.generate(dimensions, n, random_state=self.rng)

        for sample in samples:
            config = {}
            for i, param in enumerate(self.continuous_params):
                config[param] = sample[i]
            for param, choices in self.discrete_params.items():
                config[param] = self.rng.choice(choices)
            configs.append(config)
        return configs