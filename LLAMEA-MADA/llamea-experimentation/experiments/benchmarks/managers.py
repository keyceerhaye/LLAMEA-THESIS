import os
import numpy as np
import re
from openai import OpenAI
from datetime import datetime
from llamea.utils import NoCodeException

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
        os.mkdir(dirname)
        os.mkdir(f"{dirname}/ioh")
        os.mkdir(f"{dirname}/code")
        return dirname

    def log_conversation(self, content):
        """
        Logs the given conversation content into a conversation log file.
        
        Args:
            content (str): The conversation content to be logged.
        """
        with open(f"{self.dirname}/conversationlog.txt", "a") as file:
            file.write(content)

    def log_code(self, attempt, algorithm_name, code):
        """
        Logs the provided code into a file, uniquely named based on the attempt number and algorithm name.
        
        Args:
            attempt (int): The attempt number of the code execution.
            algorithm_name (str): The name of the algorithm used.
            code (str): The source code to be logged.
        """
        with open(f"{self.dirname}/code/try-{attempt}-{algorithm_name}.py", "w") as file:
            file.write(code)

    def log_aucs(self, attempt, aucs):
        """
        Logs the given AOCCs (Area Over the Convergence Curve, named here auc) into a file, named based on the attempt number.
        
        Args:
            attempt (int): The attempt number corresponding to the AOCCs.
            aucs (array_like): An array of AUC scores to be saved.
        """
        np.savetxt(f"{self.dirname}/try-{attempt}-aucs.txt", aucs)

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
        self.role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems. Do not use hyped nature-inspired algorithms such as Harmony Search, Grey Wolf, Firefly, Whale optimizer etc. since these are generally not well performing."
        self.block_guidance_prompt = (
            "For modular editing, the optimizer class must define four building blocks with meaningful logic:\n"
            "1. `parent_selection(self, population)` – decide which individuals act as parents.\n"
            "2. `recombination(self, parents)` – explain how parent information is merged.\n"
            "3. `mutation(self, candidate)` – inject stochastic variation into a candidate.\n"
            "4. `survivor_selection(self, population, offspring)` – determine how the next generation is formed.\n"
            "Do not leave these methods as placeholders; a bandit-based system will later recombine them independently."
        )
        self.init_prompt = f"""
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
{self.block_guidance_prompt}
Give an excellent and novel heuristic algorithm to solve this task and also give it a name. Give the response in the format:
# Name: <classname>
# Code: <code>
"""
        self.debug_mode = False

    def fetch_algorithm(self):
        """
        Fetches an algorithm from the AI model based on the specified role and initial prompts.
        
        Returns:
            str: The fetched algorithm code.
        """
        if self.debug_mode:
            with open("iohllm/example.txt", "r") as file:
                return file.read()

        session_messages = [
            {"role": "system", "content": self.role_prompt},
            {"role": "user", "content": self.init_prompt}
        ]
        self.logger.log_conversation(self.init_prompt)
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
        
        best_aocc_context = self.current_best_AOCC if self.current_best_AOCC > 0 else max(auc_mean, 0.0)
        if self.last_error:
            refine_prompt = (f"The last proposed algorithm {algorithm_name} got an error: {self.last_error}, "
                            f"an average Area over the convergence curve (AOCC, 1.0 is the best) of {auc_mean:.02f}, and a standard deviation of {auc_std:.02f}. "
                            f"Either refine or redesign to improve the algorithm. Give the response in the format:\n"
                            f"# Name: <classname>\n"
                            f"# Code: <code>")
        else:
            refine_prompt = (f"The last proposed algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) of {auc_mean:.02f}, "
                            f"and a standard deviation of {auc_std:.02f}. Either refine or redesign to improve the algorithm. Give the response in the format:\n"
                            f"# Name: <classname>\n"
                            f"# Code: <code>")
        mutation_directive = (
            "Mutation directive: Create a random new algorithm that treats the current best algorithm "
            f"(AOCC {best_aocc_context:.02f}) only as a reference point for performance. The new design must "
            "deliberately explore a different architecture or search strategy instead of reusing the best parent. "
            "Explicitly implement the four MADA building blocks (`parent_selection`, `recombination`, "
            "`mutation`, `survivor_selection`) with substantive logic so they can be modified independently."
        )
        refine_prompt = f"{refine_prompt}\n\n{mutation_directive}\n\n{self.block_guidance_prompt}"
            
        if self.detailed_feedback:
            detailed_feedback_prompt = (f"The mean AOCC score of the last algorithm on Separable functions was {detailed_aucs[0]:.02f}, "
                                        f"on functions with low or moderate conditioning {detailed_aucs[1]:.02f}, "
                                        f"on functions with high conditioning and unimodal {detailed_aucs[2]:.02f}, "
                                        f"on Multi-modal functions with adequate global structure {detailed_aucs[3]:.02f}, "
                                        f"and on Multi-modal functions with weak global structure {detailed_aucs[4]:.02f}")
            
        if self.elitism:
            elitism_prompt = (f"The best so far proposed algorithm got an average AOCC of {self.current_best_AOCC:.02f} and the code was as follows:\n"
                            f"{self.current_best_algorithm}")

        session_messages = [
            {"role": "system", "content": self.role_prompt},
            {"role": "user", "content": self.init_prompt},
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

    def generate_block_snippet(self, block_name: str, prompt: str) -> str:
        """
        Ask the LLM to generate or refine a specific block/method.

        Args:
            block_name (str): Name of the method (e.g., ``mutation``) to generate.
            prompt (str): Fully rendered user prompt containing context.

        Returns:
            str: Raw LLM response containing the code block.
        """

        session_messages = [
            {"role": "system", "content": self.role_prompt},
            {"role": "user", "content": prompt},
        ]
        for msg in session_messages:
            self.logger.log_conversation(msg["content"])
        call_kwargs = {
            "model": self.ai_model,
            "messages": session_messages,
            "temperature": 0.6,
        }
        if self.max_tokens:
            call_kwargs["max_tokens"] = self.max_tokens
        response = self.client.chat.completions.create(**call_kwargs)
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
            message (str): The message string containing the algorithm name and code,
                typically starting with a line like: ``# Name: Descriptive Algorithm Name``.

        Returns:
            str: Extracted algorithm name or empty string if none is found.
        """
        # Look for a plain "# Name: ..." line (no backticks), and capture the rest of the line
        # Example that this matches:
        #   # Name: Adaptive Covariance Matrix Annealing (ACMA-ES)
        pattern = r"^#\s*Name:\s*(.+)$"
        match = re.search(pattern, message, re.IGNORECASE | re.MULTILINE)
        if match:
            # Return the stripped descriptive name
            return match.group(1).strip()
        else:
            return ""

