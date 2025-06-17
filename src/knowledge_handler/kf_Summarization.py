import json
import os
import textwrap
from datetime import datetime
from knowledge_handler.gpt import GPT

class KF_Sum(GPT):
    def __init__(self, api_base, api_key, db="postgres", model=GPT.__init__.__defaults__[0]):
        super().__init__(api_base, api_key, model=model)
        self.db = db
        folder = datetime.now().strftime("%Y-%m-%d")
        os.makedirs(folder, exist_ok=True)

    def summarize(self, name, analyses):
        prompt = textwrap.dedent(f"""
            I will provide you with runtime analysis information for a database tuning knob named `{name}`.
            This analysis includes observed performance shifts, sensitivity to changes, and correlation with key metrics
            such as throughput, latency, and memory usage.

            Your task is to generate a concise performance summary for `{name}` based on the provided tuning analysis.

            Step 1: Carefully read the analysis results for `{name}` below.

            ANALYSIS RESULT:
            {analyses}

            Step 2: Write a summary that captures:
            - The role and impact of `{name}` in the tuning process.
            - Whether its adjustment significantly improved or degraded performance.
            - If possible, note any interactions with other knobs or workload patterns.
            - Any recommendation or observation that could guide future tuning.

            Return your answer in the following JSON format:
            {{
                "summary": "..."
            }}
        """)

        suggestions = self.get_GPT_response_json(prompt)
        file_path = os.path.join(self.folder, f"{name}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(suggestions["summary"], f, indent=4, ensure_ascii=False)
    
        return suggestions