
import textwrap
import json
from knowledge_handler.gpt import GPT
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, UniformIntegerHyperparameter

class KFKnobs(GPT):
    def __init__(self, api_base, api_key, db, benchmark, model=GPT.__init__.__defaults__[0]):
        super().__init__(api_base, api_key, model=model)
        self.db = db
        self.benchmark = benchmark
        if self.benchmark == 'tpch':
            self.workload = "OLAP"
        else:
            self.workload = "OLTP"
        with open('../knowledge_collection/system_description.txt', 'r', encoding='utf-8') as f:
            self.system_description = f.read()

        with open(f'../knowledge_collection/{self.db}/knob_info/system_view.json', 'r') as file:
            self.system_view = json.load(file)
        with open('../knowledge_collection/knobs.txt', 'r', encoding='utf-8') as f:
            self.candidate_knobs = [line.strip() for line in f]
        self.knob_configurable_space = ConfigurationSpace()
        for knob in self.candidate_knobs:
            if self.system_view[knob]["vartype"] == "real":
                recommended_range = self.compress(knob, "real", self.system_view[knob]["max_val"], self.system_view[knob]["min_val"], self.system_view[knob]["unit"])['recommended_range']
                self.knob_configurable_space.add_hyperparameter(UniformFloatHyperparameter(knob, recommended_range[0], recommended_range[1]))
            elif self.system_view[knob]["vartype"] == "integer":
                recommended_range = self.compress(knob, "integer", self.system_view[knob]["max_val"], self.system_view[knob]["min_val"], self.system_view[knob]["unit"])['recommended_range']
                self.knob_configurable_space.add_hyperparameter(UniformFloatHyperparameter(knob, recommended_range[0], recommended_range[1]))
            elif self.system_view[knob]["vartype"] == "enum":
                self.knob_configurable_space.add_hyperparameter(CategoricalHyperparameter(knob, self.system_view[knob]["enumvals"]))


    def compress(self, name, type, max, min, unit):
        prompt = textwrap.dedent(f"""
            I will provide you with information about a database tuning knob named `{name}`. 
            This includes its type (`{type}`), minimum value (`{min}`), maximum value (`{max}`), and unit (`{unit}`). 
            Additionally, I will describe the system context in which this knob will be tuned.

            Your task is to recommend a reasonable tuning range for `{name}` that is likely to lead to good performance, 
            taking into account the knob type, value constraints, and the characteristics of the current system.

            Step 1: Understand the knob's basic properties from the provided metadata:
            - Type: {type}
            - Min Value: {min}
            - Max Value: {max}
            - Unit: {unit}

            Step 2: Read the system description below and consider how it might influence tuning:
            SYSTEM DESCRIPTION:
            {self.system_description}

            Step 3: Recommend a sub-range [low, high] ⊆ [{min}, {max}] for tuning `{name}`. 
            The range should be practical and efficient for most workloads in this system, but need not include the entire domain. 
            Avoid extreme or unsafe values unless justified.

            Return your answer in the following JSON format:
            {{
                "recommended_range": [{min}, {max}]
            }}
            """)

        suggestions = self.get_GPT_response_json(prompt)
        return suggestions

    def selection(self, dbinfo, num):
        prompt = textwrap.dedent(f""" 
            I will provide you with two pieces of information:

            (1) `DBINFO`: A collection of descriptions for various database knobs. Each knob includes information about its purpose, effects on performance (e.g., memory, I/O, concurrency), and tuning considerations.  
            (2) `CANDIDATE_KNOBS`: A list of knob names currently considered for tuning.  
            Your task is to analyze these candidates and select the top `{num}` knobs that are most critical for performance tuning under general or expected workloads.

            Step 1: Carefully review the description of each knob in `DBINFO`. Focus on knobs that have a direct or high-impact effect on query execution, memory usage, write-ahead logging, caching, parallelism, or other core performance factors.  
            Step 2: Read the system description below and consider how it might influence tuning:
            SYSTEM DESCRIPTION:
            {self.system_description}
            Step 3: Cross-reference the `CANDIDATE_KNOBS` list with your understanding of the knob descriptions. Identify the `{num}` knobs that are most promising or influential for tuning.  
            Step 4: Output your decision in JSON format, listing the selected knob names.

            DBINFO:
            {dbinfo}

            CANDIDATE_KNOBS:
            {self.candidate_knobs}

            Return your output in the following JSON format:
            {{
                "selected_knobs": ["knob_name_1", "knob_name_2", ..., "knob_name_{num}"]
            }}
        """)

        suggestions = self.get_GPT_response_json(prompt)
        return suggestions

