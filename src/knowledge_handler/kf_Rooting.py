import textwrap
from knowledge_handler.gpt import GPT

class KF_Root(GPT):
    def __init__(self, api_base, api_key, db="postgres", model=GPT.__init__.__defaults__[0]):
        super().__init__(api_base, api_key, model=model)
        self.db = db

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
    
    def get_link_from_gpt(self, knob_name, target_name, summary, target_summary):

        
        prompt = textwrap.dedent(f"""
            I first give you information of two database knobs: the current knob `{knob_name}` and its related knob `{target_name}`. 
            Each knob is associated with a summary extracted from tuning knowledge trees. 
            Your task is to analyze their relationship and generate a concise description of their functional or semantic connection.

            Step 1: Carefully read the summary of `{knob_name}` and `{target_name}` below.

            {knob_name} SUMMARY:
            {summary}

            {target_name} SUMMARY:
            {target_summary}

            Step 2: Based on their summaries, infer whether these two knobs are functionally related (e.g., they control similar performance aspects, operate on the same memory or I/O layer, or are typically tuned together).

            Step 3: Provide a short, high-level description (1–2 sentences) of the relationship between `{knob_name}` and `{target_name}` in **JSON format**. Do not copy the summaries; instead, synthesize and abstract their connection.

            Return your answer in the following JSON format:
            {{
                "relation": "..."
            }}

            """)

        suggestions = self.get_GPT_response_json(prompt)
        return suggestions