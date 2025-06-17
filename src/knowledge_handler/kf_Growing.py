import textwrap
from knowledge_handler.gpt import GPT

class KF_Grow(GPT):
    def __init__(self, api_base, api_key, db="postgres", model=GPT.__init__.__defaults__[0]):
        super().__init__(api_base, api_key, model=model)
        self.db = db
 
    
    def get_analysis_from_gpt(self, knob_name, change, best_performance):
        prompt = textwrap.dedent(f""" 
            I will provide you with the name of a database knob `{knob_name}`, the specific change made to it (`{change}`), 
            and the best performance improvement observed relative to its default configuration (approximately {best_performance}%).

            Your task is to analyze the potential reason why such a change could lead to the observed performance gain. 
            The reasoning should be grounded in the typical effect of this knob on database system behavior, such as memory usage, I/O efficiency, query planning, concurrency, etc.

            Step 1: Recall the general purpose of the knob `{knob_name}` — what aspect of the system it controls or affects.

            Step 2: Combine this understanding with the provided change (`{change}`) to hypothesize how this change likely improves system behavior and thus performance.

            Step 3: Output your reasoning in 1–2 concise, technically grounded sentences, clearly connecting the knob’s effect and the performance gain.

            Return your output in the following JSON format:
            {{
                "analysis": "..."  // concise explanation of why this change leads to the observed improvement
            }}
        """)


        suggestions = self.get_GPT_response_json(prompt)
        return suggestions

    def get_answer_from_gpt(self, knob_name, content):
        prompt = textwrap.dedent(f"""
            I will provide you with the technical description of a database knob named `{knob_name}`. The content is extracted from the official documentation or other expert sources.
            
            Your task is to carefully read this content, and perform the following steps:

            Step 1: Summarize the key behavior and purpose of the knob in 1-2 sentences. Be concise but informative. Focus on its effect on performance, memory, concurrency, logging, or other DBMS behaviors.

            Step 2: Identify and list other knobs that are semantically or functionally related to `{knob_name}` based on the content. Only include knobs mentioned in the content or clearly implied by the described behavior (e.g., knobs that jointly control buffer sizes, write-ahead logging, etc.).

            Return your answer in the following JSON format:
            {{
                "summary": "...",          // short summary of the knob's functionality
                "related_knobs": ["...", "..."]   // list of related knobs explicitly mentioned or implied
            }}

            CONTENT:
            {content}

            Now think carefully and answer in JSON:
            """)

        suggestions = self.get_GPT_response_json(prompt)
        return suggestions