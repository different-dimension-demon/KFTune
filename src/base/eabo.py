import numpy as np
import random
from typing import List, Dict, Optional, Tuple
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from skopt.sampler import Lhs
from skopt.space import Real
from sklearn.gaussian_process.kernels import Matern
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, UniformIntegerHyperparameter
from base.bo import BayesianOptimizer


class EvolutionaryAlgorithmAdvisor:
    def __init__(
        self,
        config_encoder,
        num_objs: int = 1,
        population_size: int = 30,
        sample_ratio: float = 0.8,
        crossover_rate: float = 0.8,
        elitism: bool = True,
        random_state: int = 42,
    ):
        self.config_encoder = config_encoder
        self.num_objs = num_objs
        self.population_size = population_size
        self.sample_ratio = sample_ratio
        self.mutation_rate = crossover_rate
        self.elitism = elitism
        self.rng = np.random.RandomState(random_state)

        self.history = []
        self.generation = 0
        self.population = []
        self.fitness = []
        self.continuous_params = config_encoder.get_continuous_dim()

    
    def _convert_config_to_array(self, config: Dict) -> np.ndarray:
        hyperparameters = self.config_space.get_hyperparameters()
        return np.array([config[hp.name] for hp in hyperparameters])

    def _convert_array_to_config(self, array: np.ndarray) -> Dict:
        hyperparameters = self.config_space.get_hyperparameters()
        return {hp.name: array[i] for i, hp in enumerate(hyperparameters)}
    
    def sample_random_configs(self, n: int = 1) -> List[Dict]:
        """Generate random configurations using Latin Hypercube Sampling for continuous parameters and random sampling for discrete parameters."""
        configs = []
        
        dimensions = [Real(*self.param_bounds[param]) for param in self.continuous_params]
    
        lhs = Lhs(lhs_type="classic", criterion=None)  # or use lhs_type="centered"
        samples = lhs.generate(dimensions, n, random_state=self.rng)
    
        for sample in samples:
            config = {}
            for i, param in enumerate(self.continuous_params):
                config[param] = sample[i]
            
            for param, choices in self.discrete_params.items():
                config[param] = self.rng.choice(choices)
        
            configs.append(config)

        return configs
    
    def ask(self, n: int) -> List[Dict]:
        if self.generation == 0:
            configs = self.config_encoder.sample_random_configs(n)
        else:
            configs = self.generate_via_evolution(n)
        self.current_candidates = configs  
        return configs
    
    def tell(self, evaluated_configs: List[Dict], objectives: List[float]):
        new_evaluations = list(zip(evaluated_configs, objectives))
        for config, objective in new_evaluations:
            encoded = self.config_encoder.encode(config)
            self.population.append(encoded)
            self.fitness.append(objective)
    
        self.history.extend(new_evaluations)
        self.history.sort(key=lambda x: x[1])
        self.generation += 1

    def generate_via_evolution(self, n: int) -> List[np.ndarray]:
        """Generate n new individuals based on the current population."""
        new_individuals = []
        parents = self.history[:5]
        while len(new_individuals) < n:        
            child1, child2 = self._crossover(random.sample(parents, k=2))
            if random.random() > self.mutation_rate:
                 child1 = self._bayesian_optimization_mutation()
            if random.random() > self.mutation_rate:
                 child2 = self._bayesian_optimization_mutation()
            child1 = self.config_encoder.decode(child1)
            child2 = self.config_encoder.decode(child2)
            new_individuals.append(child1)
            new_individuals.append(child2)
        return new_individuals
    

    def _bayesian_optimization_mutation(self):
        sample_size = int(len(self.population)*0.8)
        indices = self.rng.choice(len(self.population), size=sample_size, replace=False)
        X_sample = np.array([self.population[i] for i in indices])
        y_sample = np.array([self.fitness[i] for i in indices])

        bo = BayesianOptimizer(surrogate_type="GP", acquisition_type="PI")
        return bo.suggest(X_sample, y_sample, self.continuous_params, len(self.population[0]))


    def _crossover(self, parents: List[np.ndarray]) -> List[np.ndarray]:
        parent1, parent2 = parents
        children = []
        parent1 = self.config_encoder.encode(parent1[0])
        parent2 = self.config_encoder.encode(parent2[0])
        
        continuous_dim = self.continuous_params
        
        if continuous_dim > 0:
            crossover_point = self.rng.randint(1, continuous_dim)
            child1_continuous = np.concatenate(
                [parent1[:crossover_point], parent2[crossover_point:continuous_dim]]
            )
            child2_continuous = np.concatenate(
                [parent2[:crossover_point], parent1[crossover_point:continuous_dim]]
            )
        else:
            child1_continuous = np.array([])
            child2_continuous = np.array([])
                
        child1_discrete = np.array([])
        child2_discrete = np.array([])
            
        start_idx = continuous_dim
        for choices in self.config_encoder.discrete_params.values():
            n_choices = len(choices)
            end_idx = start_idx + n_choices
                
            alpha = self.rng.rand()
            child1_discrete_part = alpha * parent1[start_idx:end_idx] + (1 - alpha) * parent2[start_idx:end_idx]
            child2_discrete_part = alpha * parent2[start_idx:end_idx] + (1 - alpha) * parent1[start_idx:end_idx]
                
            child1_discrete_part /= np.sum(child1_discrete_part)
            child2_discrete_part /= np.sum(child2_discrete_part)
                
            child1_discrete = np.concatenate([child1_discrete, child1_discrete_part])
            child2_discrete = np.concatenate([child2_discrete, child2_discrete_part])
            
            start_idx = end_idx
                
        child1 = np.concatenate([child1_continuous, child1_discrete])
        child2 = np.concatenate([child2_continuous, child2_discrete])
            
        children.append(child1)
        children.append(child2)

            
        return children

