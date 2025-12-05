import json
import os
import numpy as np
import re
import time
from pathlib import Path
import openai
from openai import OpenAI
from datetime import datetime
from llamea.utils import NoCodeException
from llamea.mada.ast_utils import (
    default_allowed_names,
    detect_population_contract_issues,
    find_undefined_names,
)

BASE_DIR = Path(__file__).resolve().parent
SCAFFOLD_TEMPLATE_PATH = BASE_DIR / "templates" / "mada_optimizer_base.py"

BLOCK_METHOD_REQUIREMENTS = """
- `def parent_selection(self, population):` — receives a list of `(candidate, fitness)` tuples and **must** destructure them (e.g., `for candidate, fitness in population`).
- `def recombination(self, parents):` — consumes the full list returned by `parent_selection`. Keep the signature unchanged and operate on `(candidate, fitness)` tuples internally.
- `def mutation(self, candidate):` — perturbs a candidate vector while keeping it within [-5, 5].
- `def survivor_selection(self, population, offspring):` — merges parents and offspring (lists of `(candidate, fitness)` tuples) and returns the next generation of the same length. Always return tuples via the `_ensure_tuple_records` helper.
Helper utilities provided by the scaffold include `_safe_index_sample`, `_clip_vector`, `_normalize_population`, `_iter_population`, and `_ensure_tuple_records` — call them instead of re-implementing risky logic.
All four methods must contain executable logic (no placeholders) and should delegate shared behavior to helper methods instead of duplicating code.
"""

PROMPT_GUARDRAILS = [
    "- Keep shared state updates inside helper methods or __init__; do not introduce module-level globals.",
    "- Do NOT add archive/arch_* structures or external archives; keep everything in population/offspring tuples.",
    "- Do NOT call population.sort() on raw tuples; destructure `(candidate, fitness)` and sort by fitness if needed.",
    "- Always keep concrete implementations of parent_selection, recombination, mutation, and survivor_selection with the required signatures.",
    "- Prefer scaffold helpers like `_safe_index_sample` and `_clip_vector` for sampling and bound handling to avoid runtime errors.",
]

TUPLE_USAGE_EXAMPLES = """
Good tuple handling:
```python
for candidate, fitness in population:
    # candidate: np.ndarray, fitness: float
    ...

def survivor_selection(self, population, offspring):
    combined = self._normalize_population(population) + self._normalize_population(offspring)
    combined.sort(key=lambda item: item[1])
    return self._ensure_tuple_records(combined[: self.population_size])
```

Bad tuple handling (avoid):
```python
for item in population:
    x = item  # item is a tuple; destructure instead

population.sort()  # sorting raw tuples without destructuring
func((candidate, fitness))  # passing tuple directly into func
```
"""

_timeout_error_cls = getattr(openai, "APITimeoutError", None)
TRANSIENT_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APIError,
    openai.InternalServerError,
)
if _timeout_error_cls:
    TRANSIENT_ERRORS = TRANSIENT_ERRORS + (_timeout_error_cls,)

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
        dirname = f"exp-{today}-{name}"
        dir_path = Path(dirname)
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "ioh").mkdir(exist_ok=True)
        (dir_path / "code").mkdir(exist_ok=True)
        return str(dir_path)

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

    def log_reward_snapshot(
        self,
        attempt,
        child_name,
        strategy,
        child_fitness,
        baseline,
        reward,
        strategy_baseline=None,
        outer_choice="mada",
        failure_kind="success",
        error="",
    ):
        """
        Persist the reward signal the bandits observed so we can audit it later.

        Args:
            attempt (int): Current API call / attempt index.
            child_name (str): Offspring identifier.
            strategy (str): Strategy label (innovation / recombination / legacy).
            child_fitness (float): Evaluated AOCC mean for the child (0 if failure).
            baseline (float): Baseline value the reward was compared against.
            reward (float): Reward actually sent to the bandits after clipping.
            error (str, optional): Failure summary if evaluation errored.
        """
        log_path = os.path.join(self.dirname, "reward-log.csv")
        needs_header = not os.path.exists(log_path)
        safe_error = (error or "").replace("\n", " | ")
        safe_name = child_name.replace(",", " ")
        safe_strategy = strategy.replace(",", " ")
        safe_outer_choice = (outer_choice or "").replace(",", " ")
        safe_failure_kind = (failure_kind or "success").replace(",", " ")
        line = (
            f"{attempt},{safe_name},{safe_strategy},"
            f"{child_fitness:.6f},{baseline:.6f},{reward:.6f},"
            f"{(strategy_baseline if strategy_baseline is not None else baseline):.6f},"
            f"{safe_outer_choice},{safe_failure_kind},{safe_error}\n"
        )
        with open(log_path, "a", encoding="utf-8") as file:
            if needs_header:
                file.write(
                    "attempt,child,strategy,child_fitness,baseline,reward,"
                    "strategy_baseline,outer_choice,failure_kind,error\n"
                )
            file.write(line)

    def log_monitoring_snapshot(self, payload: dict) -> None:
        """
        Append a JSON line containing monitoring metadata (coverage, strategies, etc.).
        """
        log_path = os.path.join(self.dirname, "monitoring.jsonl")
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(payload))
            file.write("\n")


class LLMTransientError(RuntimeError):
    """Raised when the LLM API repeatedly fails to respond."""


class InvalidAlgorithmError(ValueError):
    """Raised when extracted code references undefined identifiers."""

class AlgorithmManager:
    def __init__(
        self,
        api_key,
        logger,
        ai_model="gemini-2.0-flash",
        elitism=False,
        detailed_feedback=False,
        base_url=None,
        max_tokens=None,
        request_timeout=90.0,
    ):
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
            request_timeout (float, optional): Seconds to wait for each LLM response before retrying.
        """
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if request_timeout:
            client_kwargs["timeout"] = request_timeout
        self.client = OpenAI(**client_kwargs)
        self.ai_model = ai_model
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.elitism = elitism
        self.current_best_algorithm = ""
        self.current_best_AOCC = 0
        self.detailed_feedback = detailed_feedback
        self.logger = logger
        self.last_algorithm = ""
        self.tried_algorithms = ""
        self.last_error = ""
        self.last_traceback_tail = ""
        self.role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems. Do not use hyped nature-inspired algorithms such as Harmony Search, Grey Wolf, Firefly, Whale optimizer etc. since these are generally not well performing."
        self.dynamic_guardrail_note = ""
        self.scaffold_template = self._load_scaffold_template()
        self.allowed_identifier_names = default_allowed_names({"i", "idx"})
        self.last_algorithm = self.scaffold_template
        try:
            self.current_best_algorithm = self.extract_algorithm_code(self.scaffold_template)
        except NoCodeException:
            self.current_best_algorithm = ""
        scaffold_prompt = self._scaffold_prompt_block()
        self._base_init_prompt = f"""
The optimization algorithm should handle the BBOB test suite (24 noiseless functions, bounds [-5, 5], typical dimension 5). Your task is to provide Python code that maximizes performance under a fixed evaluation budget.

Your optimizer MUST:
1. Define `__init__(self, budget=10000, dim=5)` that stores the budget, dimensionality, and any helper state.
2. Define `def __call__(self, func):` that orchestrates the search and counts every call to `func`.
3. Provide all FOUR modular block methods with executable logic and the exact signatures listed below. They must operate on `(candidate, fitness)` tuples and may call helper methods for shared behavior.
{BLOCK_METHOD_REQUIREMENTS}
4. Keep helper utilities inside the class (e.g., `_initialize_population`, `_normalize_population`, `_ensure_tuple_records`) and call them from the block methods instead of copying code.
5. Respect the budget strictly and keep all candidate vectors within the problem bounds.

You MUST start from the canonical scaffold below and only modify its methods/helpers:
{scaffold_prompt}

Tuple-handling examples (follow GOOD, avoid BAD):
{TUPLE_USAGE_EXAMPLES}

Return format:
# Name: <classname>
# Code:
```python
<code>
```
"""
        self.debug_mode = False

    def update_guardrail_feedback(self, note: str):
        note = (note or "").strip()
        reminder = ""
        if "Undefined identifiers detected" in note or "undefined_identifiers" in note:
            reminder = "Use tuple unpacking: for candidate, fitness in population:"
        parts = [p for p in (note, reminder) if p]
        self.dynamic_guardrail_note = "; ".join(parts)

    def _guardrail_lines(self):
        lines = list(PROMPT_GUARDRAILS)
        if self.dynamic_guardrail_note:
            lines.append(f"- Recent issues observed: {self.dynamic_guardrail_note}")
        return lines

    def _render_guardrail_block(self):
        lines = self._guardrail_lines()
        if not lines:
            return ""
        return "\nPlease respect the following guardrails:\n" + "\n".join(lines) + "\n"

    def _structure_directive(self) -> str:
        helper_sentence = (
            "Use the scaffold helpers (e.g., `_safe_index_sample`, `_clip_vector`, `_normalize_population`, `_ensure_tuple_records`, `_iter_population`) to avoid runtime sampling/clipping bugs, and keep each block fully executable."
        )
        return f"{BLOCK_METHOD_REQUIREMENTS.strip()}\n{helper_sentence}"

    def _load_scaffold_template(self) -> str:
        try:
            code = SCAFFOLD_TEMPLATE_PATH.read_text(encoding="utf-8").strip()
        except OSError:
            code = (
                "import numpy as np\n\n"
                "class MADAOptimizerTemplate:\n"
                "    def __init__(self, budget=10000, dim=5):\n"
                "        self.budget = budget\n"
                "        self.dim = dim\n"
                "    def __call__(self, func):\n"
                "        return 0.0, np.zeros(1)\n"
            )
        return (
            "# Name: MADAOptimizerTemplate\n"
            "# Code:\n"
            "```python\n"
            f"{code}\n"
            "```\n"
        )

    def _scaffold_prompt_block(self) -> str:
        return self.scaffold_template or ""

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
        response = self._chat_completion(session_messages, temperature=0.8)
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
        structure_directive = (
            "While refining, keep the required block signatures intact, ensure each block contains runnable code (no placeholders), and route shared logic through helper methods."
        )
        block_requirements = self._structure_directive()
        traceback_block = ""
        if self.last_traceback_tail:
            traceback_block = (
                "\nRecent evaluation traceback (truncated):\n"
                "```python\n"
                f"{self.last_traceback_tail.strip()}\n"
                "```\n"
            )
        scaffold_prompt = self._scaffold_prompt_block()
        tuple_hint = ""
        if "candidate" in (self.last_error or "").lower() or "fitness" in (self.last_error or "").lower():
            tuple_hint = (
                "\nReminder: unpack tuples explicitly (e.g., `for candidate, fitness in population:`) and never pass `(candidate, fitness)` directly into func/mutation.\n"
                f"{TUPLE_USAGE_EXAMPLES}\n"
            )
        if self.last_error:
            refine_prompt = (
                f"The last proposed algorithm {algorithm_name} got an error: {self.last_error}, "
                f"an average Area over the convergence curve (AOCC, 1.0 is the best) of {auc_mean:.02f}, and a standard deviation of {auc_std:.02f}. "
                f"Either refine or redesign to improve the algorithm. Give the response in the format:\n"
                f"# Name: <classname>\n"
                f"# Code: <code>"
                f"{tuple_hint}"
                f"\nStart from the canonical scaffold:\n{scaffold_prompt}"
                f"{guardrail_suffix}\n"
                f"{traceback_block}"
                f"{structure_directive}\n"
                f"{block_requirements}"
            )
        else:
            refine_prompt = (
                f"The last proposed algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) of {auc_mean:.02f}, "
                f"and a standard deviation of {auc_std:.02f}. Either refine or redesign to improve the algorithm. Give the response in the format:\n"
                f"# Name: <classname>\n"
                f"# Code: <code>"
                f"{tuple_hint}"
                f"\nStart from the canonical scaffold:\n{scaffold_prompt}"
                f"{guardrail_suffix}\n"
                f"{traceback_block}"
                f"{structure_directive}\n"
                f"{block_requirements}"
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
        response = self._chat_completion(session_messages, temperature=0.8)
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
    
    def generate_block_snippet(self, block_name: str, prompt: str) -> str:
        """
        Request an updated implementation for a specific optimizer block.

        Args:
            block_name: Name of the block being mutated.
            prompt: Fully constructed prompt (already includes guardrails/context).
        """
        session_messages = [
            {"role": "system", "content": self.role_prompt},
            {"role": "user", "content": prompt},
        ]
        self.logger.log_conversation(f"[BLOCK:{block_name}]\n{prompt}")
        response = self._chat_completion(session_messages, temperature=0.8)
        message = response.choices[0].message.content
        self.logger.log_conversation(message)
        return message

    def validate_identifier_usage(
        self, code: str, allow_tutoring: bool = False, soft_undefined: bool = False
    ):
        """
        Ensure the provided code does not reference undefined identifiers.
        Returns a list of contract issues when allow_tutoring=True.
        """

        forbidden_matches = []
        for token in ("arch_cand", "arch_fit", "archive"):
            if re.search(rf"\\b{token}\\b", code):
                forbidden_matches.append(f"forbidden_archive:{token}")

        undefined = find_undefined_names(code, self.allowed_identifier_names)
        contract_issues = detect_population_contract_issues(code)

        if undefined:
            if allow_tutoring or soft_undefined:
                return [f"undefined_identifiers:{','.join(sorted(undefined))}"] + forbidden_matches + contract_issues
            joined = ", ".join(sorted(undefined))
            raise InvalidAlgorithmError(f"Undefined identifiers detected: {joined}")

        if forbidden_matches:
            if allow_tutoring:
                return forbidden_matches + contract_issues
            raise InvalidAlgorithmError(
                "Population block contract violations: " + "; ".join(forbidden_matches)
            )

        if contract_issues:
            if allow_tutoring:
                return forbidden_matches + list(contract_issues)
            raise InvalidAlgorithmError(
                "Population block contract violations: " + "; ".join(contract_issues)
            )
        return []

    def coerce_tuple_loops(self, code: str):
        """
        Heuristically repair missing tuple destructuring in for-loops over
        population-like collections. Returns (repaired_code, replacements_count).
        """
        replacements = 0

        def _fix_loop(match: re.Match) -> str:
            nonlocal replacements
            replacements += 1
            collection = match.group(2)
            return f"for candidate, fitness in {collection}:"

        patterns = [
            r"for\s+\w+\s+in\s+(population|offspring|parents)\s*:",  # single var
            r"for\s+\w+\s*,\s*\w+\s+in\s+(population|offspring|parents)\s*:",  # two vars
            r"for\s+\w+\s*,\s*\w+\s+in\s+enumerate\(\s*(population|offspring|parents)\s*\)\s*:",  # enumerate
        ]

        repaired = code
        for pat in patterns:
            repaired = re.sub(pat, _fix_loop, repaired)
        return repaired, replacements

    def _chat_completion(
        self,
        session_messages,
        temperature: float = 0.8,
        max_retries: int = 5,
        backoff_seconds: float = 5.0,
    ):
        """
        Execute a chat completion call with basic retry handling for transient API failures.

        Args:
            session_messages (list[dict]): conversation history (role/content pairs)
            temperature (float): sampling temperature for the LLM
            max_retries (int): maximum attempts before surfacing LLMTransientError
            backoff_seconds (float): base delay between retries (multiplied per attempt)
        """

        params = {
            "model": self.ai_model,
            "messages": session_messages,
            "temperature": temperature,
        }
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                return self.client.chat.completions.create(**params)
            except TRANSIENT_ERRORS as exc:
                last_error = exc
                delay = backoff_seconds * attempt
                time.sleep(delay)
            except Exception:
                # Non-transient errors should bubble up immediately so callers can handle them.
                raise

        raise LLMTransientError(
            f"LLM failed after {max_retries} attempts: {last_error}"
        )

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

