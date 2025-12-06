import os
import numpy as np
import re
import hashlib
from openai import OpenAI
from datetime import datetime
from llamea.utils import NoCodeException

PROMPT_GUARDRAILS = [
    "- Keep shared state updates inside helper methods or __init__; do not introduce module-level globals.",
    "- If the block depends on helpers such as update_archive(), call them instead of duplicating logic.",
]

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
        safe_name = re.sub(r"[\\/]+", "_", name or "")
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", safe_name)
        dirname = f"exp-{today}-{safe_name}"
        os.makedirs(dirname, exist_ok=True)
        os.makedirs(f"{dirname}/ioh", exist_ok=True)
        os.makedirs(f"{dirname}/code", exist_ok=True)
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

    def log_aucs(self, attempt, aucs):
        """
        Logs the given AOCCs (Area Over the Convergence Curve, named here auc) into a file, named based on the attempt number.
        
        Args:
            attempt (int): The attempt number corresponding to the AOCCs.
            aucs (array_like): An array of AUC scores to be saved.
        """
        with open(
            f"{self.dirname}/try-{attempt}-aucs.txt",
            "w",
            encoding="utf-8",
        ) as file:
            np.savetxt(file, aucs)

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
        self.rejected_hashes = set()
        self.role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems. Do not use hyped nature-inspired algorithms such as Harmony Search, Grey Wolf, Firefly, Whale optimizer etc. since these are generally not well performing."
        self.dynamic_guardrail_note = ""
        self._base_init_prompt = """
Implement a population-based optimizer with explicit recombination.

CRITICAL INTERFACE - MUST FOLLOW EXACTLY:
- __init__(self, budget): Only takes budget, nothing else
- __call__(self, func): Infer dimension from bounds, return (self.f_opt, self.x_opt)

MANDATORY BOUNDS HANDLING (copy this pattern exactly):
```python
def __call__(self, func):
    lb = np.asarray(func.bounds.lb)  # ALWAYS use .lb
    ub = np.asarray(func.bounds.ub)  # ALWAYS use .ub
    dim = len(lb)                     # Get dimension from lb length
    # NEVER use: func.bounds.shape, bounds[0], bounds[1], len(func.bounds)
```

WORKING EXAMPLE TO BASE YOUR CODE ON:
```python
import numpy as np

class ExampleOptimizer:
    def __init__(self, budget):
        self.budget = budget
        self.f_opt = float('inf')
        self.x_opt = None

    def __call__(self, func):
        lb = np.asarray(func.bounds.lb)
        ub = np.asarray(func.bounds.ub)
        dim = len(lb)
        pop_size = max(10, min(self.budget // 10, 50))
        
        population = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.array([func(x) for x in population])
        evals = pop_size
        
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx].copy()
        
        while evals < self.budget:
            parents = self.select_parents(population, fitness)
            offspring = self.recombine(parents)
            offspring = self.mutate(offspring, lb, ub)
            offspring = np.clip(offspring, lb, ub)
            
            offspring_f = np.array([func(x) for x in offspring])
            evals += len(offspring)
            
            population, fitness = self.survivor_selection(population, fitness, offspring, offspring_f)
            
            if fitness[0] < self.f_opt:
                self.f_opt = fitness[0]
                self.x_opt = population[0].copy()
        
        return self.f_opt, self.x_opt

    def select_parents(self, population, fitness):
        pop_size = len(population)
        tournament_size = min(3, pop_size)  # Never exceed population size
        parents = []
        for _ in range(pop_size):
            idx = np.random.choice(pop_size, tournament_size, replace=True)
            winner = idx[np.argmin(fitness[idx])]
            parents.append(population[winner])
        return np.array(parents)

    def recombine(self, parents):
        pop_size, dim = parents.shape
        offspring = np.empty_like(parents)
        for i in range(pop_size):
            p1, p2 = parents[np.random.choice(pop_size, 2, replace=False)]
            mask = np.random.rand(dim) < 0.5
            offspring[i] = np.where(mask, p1, p2)
        return offspring

    def mutate(self, offspring, lb, ub):
        mutation = np.random.normal(0, 0.1 * (ub - lb), offspring.shape)
        mask = np.random.rand(*offspring.shape) < 0.1
        return offspring + mutation * mask

    def survivor_selection(self, population, fitness, offspring, offspring_f):
        combined = np.vstack([population, offspring])
        combined_f = np.concatenate([fitness, offspring_f])
        indices = np.argsort(combined_f)[:len(population)]
        return combined[indices], combined_f[indices]
```

Create a NOVEL variation of this pattern. You must:
- Call select_parents, recombine, mutate, survivor_selection every generation
- Recombination MUST use ≥2 parents and produce different offspring
- No placeholders, TODOs, or pass statements
- Tournament size must be min(k, pop_size) to avoid sampling errors

Give the response in the format:
# Name: <classname>
# Code: <code>
"""
        self.debug_mode = False

    def update_guardrail_feedback(self, note: str):
        self.dynamic_guardrail_note = (note or "").strip()

    def _error_guardrail_lines(self):
        """Derive guardrails automatically from last_error text."""
        note = (self.last_error or "").lower()
        lines = []
        if "signatureerror" in note:
            lines.append("- __init__(self, budget) only; __call__(self, func) infers dim from func.bounds.")
            lines.append("- No extra required args; keep interfaces exactly as specified.")
        if "preflighterror" in note or "valueerror" in note or "broadcast" in note:
            lines.append("- Keep lb/ub as 1-D arrays; do not reshape/column-stack bounds.")
            lines.append("- Keep populations/parents/offspring strictly 2-D (pop_size, dim); avoid (dim,1).")
        if "attributerror" in note or "bounds" in note:
            lines.append("- Access bounds via lb = np.asarray(func.bounds.lb); ub likewise; do not use func.bounds.shape/len.")
        if "allclose" in note or "identical" in note:
            lines.append("- Ensure offspring differs from every parent (np.allclose check and resample).")
        return lines

    def _guardrail_lines(self):
        lines = list(PROMPT_GUARDRAILS)
        lines += self._error_guardrail_lines()
        if self.dynamic_guardrail_note:
            lines.append(f"- Recent issues observed: {self.dynamic_guardrail_note}")
        if self.rejected_hashes:
            recent_hashes = list(self.rejected_hashes)[-5:]
            lines.append(f"- Avoid reusing prior rejected code hashes: {', '.join(recent_hashes)}")
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for ln in lines:
            if ln in seen:
                continue
            seen.add(ln)
            deduped.append(ln)
        return deduped

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
            "temperature": 0.8
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
            "temperature": 0.8
        }
        if self.max_tokens:
            call_kwargs["max_tokens"] = self.max_tokens
        response = self.client.chat.completions.create(**call_kwargs)
        self.tried_algorithms += f"\nYou already tried {algorithm_name}, with score: {auc_mean}"
        message = response.choices[0].message.content
        self.logger.log_conversation(message)
        return message

    def record_reject(self, code: str):
        """Record a rejected code hash to steer future prompts away."""
        try:
            digest = hashlib.sha256(code.encode("utf-8")).hexdigest()[:12]
            self.rejected_hashes.add(digest)
            self.tried_algorithms += f"\nRejected code hash: {digest}"
        except Exception:
            pass

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
        # Primary: fenced code block
        pattern = r"```(?:python)?\n(.*?)\n```"
        match = re.search(pattern, message, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)

        # Fallback: HTML-like <code>...</code> blocks that some providers return
        html_match = re.search(r"<code>(.*?)</code>", message, re.DOTALL | re.IGNORECASE)
        if html_match:
            return html_match.group(1)

        # Fallback: take everything from the first 'class ' onward if present
        class_match = re.search(r"(class\s+\w+.*)", message, re.DOTALL)
        if class_match:
            return class_match.group(1).strip()

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

    def generate_block_snippet(self, block_name: str, prompt: str) -> str:
        """
        Ask the LLM to rewrite a specific block (e.g., recombination/mutation).

        The prompt should already include any contextual snippets; we append
        guardrails and keep the usual role prompt so extraction stays uniform.
        """
        guardrail_block = self._render_guardrail_block()
        suffix = f"\n{guardrail_block}" if guardrail_block else ""
        session_messages = [
            {"role": "system", "content": self.role_prompt},
            {"role": "user", "content": f"{prompt}{suffix}"},
        ]
        if self.logger and hasattr(self.logger, "log_conversation"):
            for msg in session_messages:
                try:
                    self.logger.log_conversation("\n" + msg["content"])
                except Exception:
                    pass
        call_kwargs = {
            "model": self.ai_model,
            "messages": session_messages,
            "temperature": 0.6,  # a bit steadier for block-level edits
        }
        if self.max_tokens:
            call_kwargs["max_tokens"] = self.max_tokens
        response = self.client.chat.completions.create(**call_kwargs)
        message = response.choices[0].message.content
        if self.logger and hasattr(self.logger, "log_conversation"):
            try:
                self.logger.log_conversation(message)
            except Exception:
                pass
        return message

