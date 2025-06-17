from base.default_space import DefaultSpace
from base.eabo import EvolutionaryAlgorithmAdvisor
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import shap

class Sample_Workshop(DefaultSpace):

    def __init__(self, dbms, knowledge_forest, config_encoder, test, timeout,  seed):
        self.max_iterations = 20
        if test == "tpch":
            self.weight = 1.0
        elif test == "chbenchmark":
            self.weight = 0.5
        else:
            self.weight = 0.0
        self.config_encoder = config_encoder
        self.ea = EvolutionaryAlgorithmAdvisor(
            config_encoder=config_encoder,
            population_size=15, 
            sample_ratio=0.7,    
            crossover_rate=0.8, 
            elitism=True,
            random_state=42  
        )
        self.knowledge_forest = knowledge_forest
        
        super().__init__(dbms, test, timeout, seed)

    def optimize(self):
        best_improvement = 0.0
        for iteration in range(self.max_iterations):
            configs = self.ea.ask(5)
            performances = []
            best_performance = 0.0
            for i in range(len(configs)):
                performance = self.set_and_replay(configs[i], self.weight)
                if performance > best_performance:
                    best_performance = performance
                performances.append(performance)
            
            self.ea.tell(configs, performances) 
            if best_performance - best_improvement > 0.05:
                indexs, encoded = self.shap()
                knobs = self.config_encoder.decode_index(indexs, encoded)
                for key, value in knobs:
                    self.knowledge_forest.add_grow_node(key, value, best_performance)
        self.config_encoder.best_config, _ = self.ea.history[0]
        print(self.ea.history[0])
    
    def get_history(self):
        return self.ea.population, self.ea.fitness
    
    def get_history2(self):
        return self.ea.history
    
    def shap(self):
        X = np.array(self.ea.population)
        y = np.array(self.ea.fitness)
        model = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True)
        model.fit(X, y)
        explainer = shap.KernelExplainer(model.predict, X)
        max_y_idx = np.argmax(y)
        x_target = X[max_y_idx]

        shap_values = explainer.shap_values(x_target)
        shap_values = np.array(shap_values)
        top_2_idx = np.argsort(np.abs(shap_values))[-2:][::-1]
        return top_2_idx, x_target
    

    
