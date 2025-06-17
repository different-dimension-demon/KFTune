from base.default_space import DefaultSpace
from base.bo import BayesianOptimizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import shap
import numpy as np

class Recommender(DefaultSpace):

    def __init__(self, dbms, test, timeout, config_encoder, knowledge_forest, gpt_knobs, history, seed):
        if test == "tpch":
            self.weight = 1.0
        elif test == "chbenchmark":
            self.weight = 0.5
        else:
            self.weight = 0.0
        self.max_iterations = 80
        # 定义配置空间（包含连续和离散参数）
        self.config_encoder = config_encoder
        self.history = []
        self.population, self.fitness = history
        self.continuous_params = config_encoder.get_continuous_dim()
        self.knowledge_forest = knowledge_forest
        self.gpt_knobs = gpt_knobs
        
        self.bo = BayesianOptimizer(surrogate_type="GP", acquisition_type="PI")
        super().__init__(dbms, test, timeout, seed)

    def optimize(self):
        best_performance  = np.max(self.fitness)
        for iteration in range(self.max_iterations):
            next = self.bo.suggest(self.population, self.fitness, self.continuous_params, len(self.population[0]))
            config = self.config_encoder.decode(next)
            base_config = self.config_encoder.best_config
            for knob in self.config_encoder.selected_knobs:
                base_config[knob] = config[knob]
            
            performance = self.set_and_replay(base_config, self.weight)
            if performance > best_performance:
                self.config_encoder.best_config = base_config
            next = self.config_encoder.encode(base_config)
            self.population.append(next)
            self.fitness.append(performance)
            if performance - best_performance > 0.05:
                print("Knowledge renew")
                indexs, encoded = self.shap(next)
                knobs = self.config_encoder.decode_index(indexs, encoded)
                for key, value in knobs:
                    self.knowledge_forest.add_grow_node(key, value, best_performance)

            self.history.append((config, performance))
            self.history.sort(key=lambda x: x[1])

            # Knowledge tree update
            if iteration%20 == 0 and iteration != 0:
                query_text =  "Which database knobs contribute most to system performance under different workloads?"
                results = self.knowledge_forest.query(query_text, k=2)
                selected_knobs = self.gpt_knobs.selection(results,2)['selected_knobs']
                self.config_encoder.selected_knobs = selected_knobs       

        print(self.history[0])

    def get_history(self):
        return self.population, self.fitness
    
    def get_history2(self):
        return self.history
    
    def shap(self, next):
        X = np.array(self.population)
        # X = self.ea.population
        y = np.array(self.fitness)
        model = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True)
        model.fit(X, y)
        explainer = shap.KernelExplainer(model.predict, X)

        shap_values = explainer.shap_values(next)
        shap_values = np.array(shap_values)  # shape = (num_features,)
        top_2_idx = np.argsort(np.abs(shap_values))[-2:][::-1]  # 从大到小排列取前两个索引
        return top_2_idx
    

    
