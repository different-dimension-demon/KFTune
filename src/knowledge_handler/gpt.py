from openai import OpenAI, APIError
import re
import json
import tiktoken
import textwrap

class GPT:
    def __init__(self, api_base, api_key, model="gpt-4o-mini"):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model

    def get_GPT_response_json(self, prompt, json_format=True, AAW_mode=True):
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        # === Single Interaction Round ===
        if not AAW_mode:
            if json_format:
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You should output JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model,
                    response_format={"type": "json_object"},
                    temperature=0.5,
                )
                ans = response.choices[0].message.content
                return json.loads(ans)
            else:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    temperature=1,
                )
                return response.choices[0].message.content

        # === Ask-and-Answer Workflow Mode ===
        else:
            # Step 1: Manager receives the request and generates sub-tasks
            manager_prompt = self.construct_manager_prompt(prompt)
            manager_response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are the tuning Manager. Output a JSON that decomposes the task into expert domains."},
                    {"role": "user", "content": manager_prompt}
                ],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.5,
            )
            task_plan = json.loads(manager_response.choices[0].message.content)

            # Step 2: Dispatch sub-tasks to each expert
            expert_outputs = {}
            for expert in ["Memory", "I/O", "Query Plan", "CPU", "Index", "Workload"]:
                if expert in task_plan:
                    sub_prompt = self.construct_expert_prompt(expert, task_plan[expert])
                    response = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": f"You are an expert in {expert}. Output JSON."},
                            {"role": "user", "content": sub_prompt}
                        ],
                        model=self.model,
                        response_format={"type": "json_object"},
                        temperature=0.5,
                    )
                    expert_outputs[expert] = json.loads(response.choices[0].message.content)

            # Step 4: Conflict detection and aggregated summarization
            aggregation_prompt = self.construct_aggregation_prompt(expert_outputs, prompt)
            summary_response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a summarizer. Output final tuning recommendation in JSON."},
                    {"role": "user", "content": aggregation_prompt}
                ],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.5,
            )
            final_output = json.loads(summary_response.choices[0].message.content)

            return final_output

    def construct_manager_prompt(self, original_prompt):
        return f"""
        You are the **Tuning Manager Agent** in a database knob optimization system. Your responsibility is to read the following tuning request and intelligently **delegate sub-tasks** to the appropriate expert agents based on the nature of the request.

        There are six available domain experts:
        - **Memory**: Handles buffer sizes, cache policies, and memory utilization knobs.
        - **I/O**: Covers disk I/O behaviors, WAL settings, and flushing mechanisms.
        - **Query Plan**: Analyzes execution plans, join strategies, and optimizer-relevant knobs.
        - **CPU**: Focuses on parallelism, worker processes, and CPU-related tuning.
        - **Index**: Deals with index creation, maintenance, and relevant cost knobs.
        - **Workload**: Understands the workload pattern (OLTP/OLAP/HTAP) and adapts tuning accordingly.

        ###  Your Task
        1. Carefully analyze the following **tuning request**:
        2. Decide which expert(s) are relevant to this task.
        3. For each selected expert, generate a **short sub-task prompt** that clearly communicates its role and expected contribution.

        ### Return Format
        Please output the result in the following **JSON format**, including only the relevant experts:

        ```json
        {{
            "Memory": "...sub-task prompt...",
            "CPU": "...sub-task prompt...",
            ...
        }}
        """

    def construct_expert_prompt(self, expert, task_description):
        return f"""
        As the {expert} expert, focus on the aspects of the system most relevant to your domain (e.g., memory sizing, I/O behavior, parallelism, indexing, etc.). Consider best practices, contextual clues, and tuning heuristics to form your answer.
        Analyze the following task and provide a JSON-formatted tuning recommendation.

        Task:
        {task_description}

        Output format:
        {{
            "recommendation": "...",
            "rationale": "..."
        }}
        """

    def construct_aggregation_prompt(self, expert_outputs, prompt):
        return f"""
        You are the final summarizer in an expert-driven tuning assistant system.

        You will receive tuning suggestions from multiple domain-specific experts (e.g., Memory, CPU, Index, etc.). 
        Each expert independently proposes a tuning recommendation and provides a rationale.

        Your task is to synthesize these expert responses into a **coherent and conflict-free summary**.

        Please follow these steps:

        ### Step 1: Conflict Detection
        - Carefully read all expert outputs.
        - Identify any **conflicts**, such as:
        - Contradictory recommendations for the same knob.
        - Incompatible assumptions (e.g., one expert assumes high memory, another assumes limited memory).
        - Overlapping domains (e.g., CPU and I/O both suggesting different buffer strategies).

        ### Step 2: Conflict Resolution
        - Resolve conflicts by:
        - Identifying which expert is more relevant for the affected knob.
        - Merging compatible reasoning where possible.
        - Discarding outlier suggestions if poorly justified.

        ### Step 3: Unified Summary
        - Generate a **concise** and **actionable** summary that integrates insights from all experts.
        - Avoid redundancy and maintain consistency.
        - Output only one field: "summary" (as a single-paragraph string).

        ### Expert Outputs:
        {json.dumps(expert_outputs, indent=2)}

        Original Prompt:
        {prompt}
        """