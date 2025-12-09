import os
import json
import numpy as np
import re
from openai import OpenAI
from datetime import datetime
from llamea.utils import NoCodeException

PROMPT_GUARDRAILS = [
    "- You MAY add small helper utilities and restart logic inside the class; keep them class-scoped (no module-level globals).",
    "- If the block depends on helpers such as update_archive(), call them instead of duplicating logic.",
    "- Do not introduce orphaned dependencies; use helpers already present in the Context Block.",
    "- Preserve class structure: only modify the requested block, not __init__ or __call__ wiring.",
]

# Block-level innovation prompt template (MADA 4.0 Section 5.6)
BLOCK_INNOVATION_PROMPT = """You are improving a single method block of an optimizer class.

CRITICAL REQUIREMENTS:
1. Use the EXACT method signature provided - do not change parameters or return type.
2. **ONLY use self.* attributes listed in the Context Block below.** Any unlisted self.* attribute will cause a fatal error.
3. You MAY define nested/local helper functions INSIDE the method body (these are safe).
4. Do NOT call self.* methods that aren't in the Context Block - they don't exist.
5. You MAY adjust other helper blocks minimally if required to keep the recombination/innovation coherent, but keep the public API unchanged.
6. Return ONLY the method definition inside a Python code block.
7. The method must be syntactically valid Python that integrates with the class.

**FORBIDDEN - These common attribute names DO NOT EXIST in this class:**
- self.domain, self.domain_range, self.bounds → USE: self.lb and self.ub (lower/upper bounds)
- self.population → USE: self.pop or self.archive
- self.lower_bound, self.upper_bound → USE: self.lb and self.ub
- self.n_dim, self.dimensions → USE: self.dim

{guardrails}

Class: {class_name}
Block to improve: {block_name}
Signature: {signature}

Context Block (available helpers and state):
{context}

**STRICT ATTRIBUTE CHECK - You may ONLY use these self.* attributes:**
{allowed_attributes}

Current implementation for reference:
```python
{baseline}
```

Population summary:
{population_summary}

Provide an improved implementation that ONLY uses the allowed attributes listed above.
Return ONLY the method code inside ```python ... ``` tags.
"""

# Whole-class holistic refinement prompt (MADA 5.0)
HOLISTIC_REFINEMENT_PROMPT = """You are refining an optimizer class holistically.

Target block: {block_name}
Instruction: {instruction}
Delta summary between parents: {delta_summary}

CRITICAL REQUIREMENTS:
1) Keep the SAME class name and __call__ signature; do not change the public API.
2) Modify ONLY the target block; keep other blocks functionally equivalent unless helper renames are required.
3) You MAY add helper methods or __init__ attributes needed by the new target block.
4) Preserve imports unless a change is required for the target logic.
5) Do NOT introduce module-level globals; keep state on self.*.
6) Return the FULL class code inside a Python code block.

Context summary:
{guardrails}
{population_summary}

Parent A code (base to rewrite):
```python
{parent_a_code}
```

{parent_b_section}

Return ONLY the full class code inside ```python ... ``` tags.
"""

# Simpler recombination-specific prompt (LLaMEA-BO style concept merging)
RECOMBINATION_PROMPT_SIMPLE = """You are recombining two parent optimizers.

Goal: Blend the useful behaviors of both parents into a single coherent block.
Instruction: "Combine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it."

CRITICAL REQUIREMENTS:
- Keep the SAME method signature: {signature}
- Use ONLY the self.* attributes listed in the Context Block.
- You MAY minimally adjust other helper blocks to keep recombination consistent; do not change the public API.
- Do NOT invent new self.* attributes or helpers.
- Return ONLY the method definition inside a Python code block.

Context Block (available helpers and state):
{context}

Allowed self.* attributes:
{allowed_attributes}

Parent A (higher fitness) snippet for this block:
```python
{parent_a}
```

Parent B snippet for this block:
```python
{parent_b}
```

Write the new recombination block now. Keep it concise and executable.
"""

# Semantic linter prompt for conflict resolution (MADA 4.0 Section 5.6)
SEMANTIC_LINTER_PROMPT = """You are a code linter resolving naming conflicts in merged helper functions.

The following Python class has been assembled from multiple parent algorithms. 
Some helper methods or attributes may have naming conflicts or be unused.

TASKS:
1. Rename any conflicting helper methods to unique names (append _v1, _v2, etc.)
2. Update all call sites to use the renamed methods.
3. Remove any helper methods that are never called.
4. Do NOT alter the core algorithmic logic - only fix naming and remove dead code.

Merged class code:
```python
{code}
```

Return the cleaned-up class code inside ```python ... ``` tags.
Only output the corrected code, no explanations.
"""

class ExperimentLogger:
    def __init__(self, name=""):
        """
        Initializes an instance of the ExperimentLogger.
        Sets up a new logging directory named with the current date and time.

        Args:
            name (str): The name of the experiment.
        """
        self.dirname = self.create_log_dir(name)
    
    def create_log_dir(self, name=""):
        """
        Creates a new directory for logging experiments based on the current date and time.
        Also creates subdirectories for IOH experimenter data and code files.
        
        Returns:
            str: The name of the created directory.
        """
        today = datetime.today().strftime('%m-%d_%H%M%S')
        # Sanitize experiment name to avoid path separator issues (e.g., model names like "zhipu/glm-4.5")
        safe_name = re.sub(r"[\\/]+", "-", name.strip()) if name else ""
        dirname = f"exp-{today}-{safe_name}" if safe_name else f"exp-{today}"

        # Create nested directories safely
        os.makedirs(dirname, exist_ok=True)
        os.makedirs(os.path.join(dirname, "ioh"), exist_ok=True)
        os.makedirs(os.path.join(dirname, "code"), exist_ok=True)
        return dirname

    def log_conversation(self, content):
        """
        Logs the given conversation content into a conversation log file.
        
        Args:
            content (str): The conversation content to be logged.
        """
        with open(f"{self.dirname}/conversationlog.txt", "a", encoding="utf-8") as file:
            file.write(content)

    def log_code(self, attempt, algorithm_name, code):
        """
        Logs the provided code into a file, uniquely named based on the attempt number and algorithm name.
        
        Args:
            attempt (int): The attempt number of the code execution.
            algorithm_name (str): The name of the algorithm used.
            code (str): The source code to be logged.
        """
        with open(
            f"{self.dirname}/code/try-{attempt}-{algorithm_name}.py",
            "w",
            encoding="utf-8",
        ) as file:
            file.write(code)

    def log_aucs(self, attempt, aucs, metadata=None):
        """
        Logs the given AOCCs (Area Over the Convergence Curve, named here auc) into a file, named based on the attempt number.
        
        Args:
            attempt (int): The attempt number corresponding to the AOCCs.
            aucs (array_like): An array of AUC scores to be saved.
            metadata (dict, optional): Optional summary info to embed as header (len/mean/std/hash).
        """
        header = ""
        if metadata:
            try:
                # Compact one-line header for quick inspection.
                parts = [f"{k}={v}" for k, v in metadata.items()]
                header = " ".join(parts)
            except Exception:
                header = ""
        with open(
            f"{self.dirname}/try-{attempt}-aucs.txt",
            "w",
            encoding="utf-8",
        ) as file:
            np.savetxt(file, aucs, header=header)

    def log_mada_offspring(self, attempt, offspring_record):
        """
        Logs MADA offspring metadata including strategy, bandit decisions, reward, and lineage.
        
        Args:
            attempt (int): The attempt/offspring number.
            offspring_record (dict): Dictionary containing:
                - strategy: str (innovation/recombination/legacy)
                - decisions: list of bandit arm selections per block
                - reward: float (fitness delta)
                - placeholder_ratio: float
                - parent_ids: list of parent UUIDs
        """
        mada_log_path = f"{self.dirname}/mada_offspring.jsonl"
        record = {"attempt": attempt, **offspring_record}
        with open(mada_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_bandit_snapshot(self, generation, bandit_states):
        """
        Logs a snapshot of all DS-TS bandit states at a given generation.
        
        Args:
            generation (int): Current generation number.
            bandit_states (dict): Dictionary mapping block names to their arm statistics.
        """
        snapshot_path = f"{self.dirname}/bandit_snapshots.jsonl"
        record = {"generation": generation, "bandits": bandit_states}
        with open(snapshot_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

class AlgorithmManager:
    def __init__(self, api_key, logger, ai_model="gemini-2.0-flash", elitism=False, detailed_feedback=False, base_url=None, max_tokens=None):
        """
        Initializes an instance of AlgorithmManager, which calls the GPT API and specifies the prompts.
        
        Args:
            api_key (str): API key for accessing OpenAI services.
            logger (object): Logger instance for recording activity and results.
            ai_model (str): The model of AI to use, defaults to 'gpt-4-turbo'.
            elitism (bool): Flag to enable using only the best algorithm from previous attempts.
            detailed_feedback (bool): Flag to enable detailed feedback about algorithm performance.
            base_url (str, optional): Custom base URL for OpenAI-compatible APIs.
            max_tokens (int, optional): Maximum tokens for API responses.
        """
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.ai_model = ai_model
        self.max_tokens = max_tokens
        self.elitism = elitism
        self.current_best_algorithm = ""
        self.current_best_AOCC = 0
        self.detailed_feedback = detailed_feedback
        self.logger = logger
        self.last_algorithm = ""
        self.tried_algorithms = ""
        self.last_error = ""
        self.role_prompt = (
            "You are a highly skilled computer scientist in the field of natural computing. "
            "Your task is to design novel metaheuristic algorithms to solve black box optimization problems. "
            "Do not use hyped nature-inspired algorithms such as Harmony Search, Grey Wolf, Firefly, Whale optimizer etc. since these are generally not well performing. "
            "You may introduce concise helper utilities, restart logic, and archive/step-size management inside the class, provided the public __init__ and __call__ stay intact and total budget is respected."
        )
        self.dynamic_guardrail_note = ""
        self._base_init_prompt = """
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality is set to 5.
An example of such code (a simple random search), is as follows:
```
class RandomSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = None

    def __call__(self, func):
        if self.dim is None:
            self.dim = len(func.bounds.lb)
        self.f_opt = np.Inf
        self.x_opt = None
        for i in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            
        return self.f_opt, self.x_opt
```
Give an excellent and novel heuristic algorithm to solve this task and also give it a name. Give the response in the format:
# Name: <classname>
# Code: <code>
"""
        self.debug_mode = False

    def update_guardrail_feedback(self, note: str):
        self.dynamic_guardrail_note = (note or "").strip()

    def _guardrail_lines(self):
        lines = list(PROMPT_GUARDRAILS)
        # Encourage diversification away from DE-style moves for new seeds.
        lines.append("- Prefer alternative search moves over classic DE-style mutation/crossover when possible.")
        if self.dynamic_guardrail_note:
            lines.append(f"- Recent issues observed: {self.dynamic_guardrail_note}")
        return lines

    def _render_guardrail_block(self):
        lines = self._guardrail_lines()
        if not lines:
            return ""
        return "\nPlease respect the following guardrails:\n" + "\n".join(lines) + "\n"

    def _build_init_prompt(self):
        guardrail_block = self._render_guardrail_block()
        if guardrail_block:
            return f"{self._base_init_prompt}\n{guardrail_block}"
        return self._base_init_prompt

    def fetch_algorithm(self):
        """
        Fetches an algorithm from the AI model based on the specified role and initial prompts.
        
        Returns:
            str: The fetched algorithm code.
        """
        if self.debug_mode:
            with open("iohllm/example.txt", "r") as file:
                return file.read()

        init_prompt = self._build_init_prompt()
        session_messages = [
            {"role": "system", "content": self.role_prompt},
            {"role": "user", "content": init_prompt}
        ]
        self.logger.log_conversation(init_prompt)
        call_kwargs = {
            "model": self.ai_model,
            "messages": session_messages,
            "temperature": 0.8,
        }
        if self.max_tokens:
            call_kwargs["max_tokens"] = self.max_tokens
        response = self.client.chat.completions.create(**call_kwargs)
        message = response.choices[0].message.content
        self.logger.log_conversation(message)
        return message

    def refine_algorithm(self, auc_mean, auc_std, algorithm_name, detailed_aucs):
        """
        Refines or redesigns an algorithm based on its performance metrics and error details.
        
        Args:
            auc_mean (float): Mean Area Over the Convergence Curve (AOCC) of the algorithm.
            auc_std (float): Standard deviation of the AOCC.
            algorithm_name (str): Name of the algorithm to refine.
            detailed_aucs (list): Detailed AOCC scores across various function types.
        
        Returns:
            str: The refined or new algorithm proposal.
        """
        if auc_mean > self.current_best_AOCC:
            self.current_best_AOCC = auc_mean
            self.current_best_algorithm = self.extract_algorithm_code(self.last_algorithm)
        
        guardrail_block = self._render_guardrail_block()
        guardrail_suffix = f"\n{guardrail_block}" if guardrail_block else ""
        if self.last_error:
            refine_prompt = (
                f"The last proposed algorithm {algorithm_name} got an error: {self.last_error}, "
                f"an average Area over the convergence curve (AOCC, 1.0 is the best) of {auc_mean:.02f}, and a standard deviation of {auc_std:.02f}. "
                f"Either refine or redesign to improve the algorithm. Give the response in the format:\n"
                f"# Name: <classname>\n"
                f"# Code: <code>"
                f"{guardrail_suffix}"
            )
        else:
            refine_prompt = (
                f"The last proposed algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) of {auc_mean:.02f}, "
                f"and a standard deviation of {auc_std:.02f}. Either refine or redesign to improve the algorithm. Give the response in the format:\n"
                f"# Name: <classname>\n"
                f"# Code: <code>"
                f"{guardrail_suffix}"
            )
            
        if self.detailed_feedback:
            detailed_feedback_prompt = (f"The mean AOCC score of the last algorithm on Separable functions was {detailed_aucs[0]:.02f}, "
                                        f"on functions with low or moderate conditioning {detailed_aucs[1]:.02f}, "
                                        f"on functions with high conditioning and unimodal {detailed_aucs[2]:.02f}, "
                                        f"on Multi-modal functions with adequate global structure {detailed_aucs[3]:.02f}, "
                                        f"and on Multi-modal functions with weak global structure {detailed_aucs[4]:.02f}")
            
        if self.elitism:
            elitism_prompt = (f"The best so far proposed algorithm got an average AOCC of {self.current_best_AOCC:.02f} and the code was as follows:\n"
                            f"{self.current_best_algorithm}")

        init_prompt = self._build_init_prompt()
        session_messages = [
            {"role": "system", "content": self.role_prompt},
            {"role": "user", "content": init_prompt},
            {"role": "user", "content": self.tried_algorithms},
            {"role": "assistant", "content": self.last_algorithm},
        ]
        if self.detailed_feedback:
            session_messages.append({"role": "user", "content": detailed_feedback_prompt})
        if self.elitism:
            session_messages.append({"role": "user", "content": elitism_prompt})
        session_messages.append({"role": "user", "content": refine_prompt})
        #add tried algorithm for next time.
        for msg in session_messages:
            self.logger.log_conversation("\n"+msg["content"])
        call_kwargs = {
            "model": self.ai_model,
            "messages": session_messages,
            "temperature": 0.8,
        }
        if self.max_tokens:
            call_kwargs["max_tokens"] = self.max_tokens
        response = self.client.chat.completions.create(**call_kwargs)
        self.tried_algorithms += f"\nYou already tried {algorithm_name}, with score: {auc_mean}"
        message = response.choices[0].message.content
        self.logger.log_conversation(message)
        return message

    def extract_algorithm_code(self, message):
        """
        Extracts algorithm code from a given message string using regular expressions.
        
        Args:
            message (str): The message string containing the algorithm code.

        Returns:
            str: Extracted algorithm code.

        Raises:
            NoCodeException: If no code block is found within the message.
        """
        pattern = r"```(?:python)?\n(.*?)\n```"
        match = re.search(pattern, message, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
        else:
            raise NoCodeException
    
    def extract_algorithm_name(self, message):
        """
        Extracts algorithm name from a given message string using regular expressions.
        
        Args:
            message (str): The message string containing the algorithm name and code.

        Returns:
            str: Extracted algorithm name or empty string.
        """
        pattern = r"`#\s*Name:\s*(\\w*)`"
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1)
        else:
            return ""

    def generate_block_snippet(self, block_name, prompt):
        """
        Generates a single method block using the LLM (MADA 4.0 block-level innovation).
        
        Args:
            block_name (str): Name of the block being generated (e.g., 'mutation', 'recombination').
            prompt (str): Full prompt including context, signature, and instructions.
        
        Returns:
            str: The LLM response containing the generated block code.
        
        Raises:
            NoCodeException: If no code block is found in the response.
        """
        session_messages = [
            {"role": "system", "content": self.role_prompt},
            {"role": "user", "content": prompt}
        ]
        
        self.logger.log_conversation(f"\n[MADA Block Generation: {block_name}]\n{prompt}")
        
        call_kwargs = {
            "model": self.ai_model,
            "messages": session_messages,
            "temperature": 0.7,  # Slightly lower for more focused block generation
        }
        if self.max_tokens:
            call_kwargs["max_tokens"] = self.max_tokens
        
        response = self.client.chat.completions.create(**call_kwargs)
        message = response.choices[0].message.content
        
        self.logger.log_conversation(f"\n[MADA Block Response: {block_name}]\n{message}")
        
        return message

    def run_semantic_linter(self, code, timeout_fallback=None):
        """
        Runs the semantic linter to resolve naming conflicts and remove unused helpers (MADA 4.0).
        
        Args:
            code (str): The merged class code with potential conflicts.
            timeout_fallback (str, optional): Fallback code to return if linting fails.
        
        Returns:
            str: Cleaned-up code with conflicts resolved.
        """
        prompt = SEMANTIC_LINTER_PROMPT.format(code=code)
        
        session_messages = [
            {"role": "system", "content": "You are a precise code refactoring assistant."},
            {"role": "user", "content": prompt}
        ]
        
        self.logger.log_conversation("\n[MADA Semantic Linter Request]\n")
        
        try:
            call_kwargs = {
                "model": self.ai_model,
                "messages": session_messages,
                "temperature": 0.2,  # Low temperature for deterministic linting
            }
            if self.max_tokens:
                call_kwargs["max_tokens"] = self.max_tokens
            
            response = self.client.chat.completions.create(**call_kwargs)
            message = response.choices[0].message.content
            
            self.logger.log_conversation(f"\n[MADA Semantic Linter Response]\n{message}")
            
            # Extract the cleaned code
            cleaned_code = self.extract_algorithm_code(message)
            return cleaned_code
            
        except Exception as e:
            self.logger.log_conversation(f"\n[MADA Semantic Linter Error: {e}]\n")
            return timeout_fallback if timeout_fallback else code

    def build_semantic_refinement_prompt(
        self,
        block_name: str,
        parent_a_code: str,
        population_summary: str = "",
        instruction: str = "",
        parent_b_code: str | None = None,
        guardrails: str | None = None,
        delta_summary: str | None = None,
    ) -> str:
        """
        Build a whole-class refinement prompt (MADA 5.0 holistic semantic rewrite).
        """
        guardrail_block = guardrails or self._render_guardrail_block()
        guardrail_text = guardrail_block if guardrail_block else ""
        parent_b_section = ""
        if parent_b_code:
            parent_b_section = (
                "Parent B code (for semantic transfer):\n```python\n"
                f"{parent_b_code}\n```"
            )
        inst = instruction or (
            f"Rewrite the {block_name} logic with a stronger strategy. Keep wiring intact."
        )
        delta_text = delta_summary or "(no delta summary provided)"
        return HOLISTIC_REFINEMENT_PROMPT.format(
            block_name=block_name,
            instruction=inst,
            guardrails=guardrail_text,
            population_summary=population_summary or "(no population summary provided)",
            parent_a_code=parent_a_code,
            parent_b_section=parent_b_section,
            delta_summary=delta_text,
        )

    def generate_semantic_refinement(self, block_name: str, prompt: str):
        """
        Generate a whole-class refinement using the LLM (returns full class code).
        """
        session_messages = [
            {"role": "system", "content": self.role_prompt},
            {"role": "user", "content": prompt},
        ]

        self.logger.log_conversation(f"\n[MADA Holistic Refinement: {block_name}]\n{prompt}")

        call_kwargs = {
            "model": self.ai_model,
            "messages": session_messages,
            "temperature": 0.7,
        }
        if self.max_tokens:
            call_kwargs["max_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(**call_kwargs)
        message = response.choices[0].message.content

        self.logger.log_conversation(f"\n[MADA Holistic Response: {block_name}]\n{message}")

        return message

    def build_block_innovation_prompt(
        self,
        block_name,
        class_name,
        signature,
        baseline,
        context,
        population_summary="",
        allowed_attributes=None,
        parent_a_snippet=None,
        parent_b_snippet=None,
    ):
        """
        Builds the prompt for block-level innovation (MADA 4.0 Section 5.2).
        
        Args:
            block_name (str): Name of the block to improve.
            class_name (str): Name of the optimizer class.
            signature (str): Method signature to preserve.
            baseline (str): Current implementation of the block.
            context (str): Context block info (helpers, attributes, imports).
            population_summary (str): Summary of current population for guidance.
            parent_a_snippet (str, optional): Higher-fitness parent block (for recombination).
            parent_b_snippet (str, optional): Other parent block (for recombination).
            allowed_attributes (list, optional): List of allowed self.* attributes.
        
        Returns:
            str: Formatted prompt for block innovation.
        """
        guardrails = self._render_guardrail_block()
        
        # Format allowed attributes as a bullet list for clarity
        if allowed_attributes:
            attr_list = "\n".join(f"  - self.{attr}" for attr in sorted(allowed_attributes))
        else:
            attr_list = "  (No specific attributes detected - use only those in __init__)"

        # If this is a recombination block and both parent snippets are provided, use the simpler BO-style prompt
        if block_name == "recombination" and parent_a_snippet and parent_b_snippet:
            return RECOMBINATION_PROMPT_SIMPLE.format(
                signature=signature,
                context=context,
                allowed_attributes=attr_list,
                parent_a=parent_a_snippet,
                parent_b=parent_b_snippet,
            )

        # Default block-level innovation prompt
        return BLOCK_INNOVATION_PROMPT.format(
            guardrails=guardrails,
            class_name=class_name,
            block_name=block_name,
            signature=signature,
            context=context,
            baseline=baseline,
            population_summary=population_summary,
            allowed_attributes=attr_list,
        )

