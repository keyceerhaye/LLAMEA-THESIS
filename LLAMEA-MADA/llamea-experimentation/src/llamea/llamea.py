"""LLaMEA - LLM powered Evolutionary Algorithm for code optimization
This module integrates OpenAI's language models to generate and evolve
algorithms to automatically evaluate (for example metaheuristics evaluated on BBOB).
"""

import concurrent.futures
import contextlib
import logging
import os
import random
import re
import traceback
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from ConfigSpace import ConfigurationSpace
from joblib import Parallel, delayed

from .loggers import ExperimentLogger
from .mada import BlockParser, DiscountedThompsonSampler
from .prompts import CROSSOVER_PURE, MUTATION_EXPLORE, MUTATION_SIMPLIFY
from .solution import Solution
from .utils import (
    DiscountedThompsonSampling,
    NoCodeException,
    code_distance,
    discrete_power_law_distribution,
    handle_timeout,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class LLaMEA:
    """
    A class that represents the Language Model powered Evolutionary Algorithm (LLaMEA).
    This class handles the initialization, evolution, and interaction with a language model
    to generate and refine algorithms.
    """

    def __init__(
        self,
        f,
        llm,
        n_parents=5,
        n_offspring=5,
        role_prompt="",
        task_prompt="",
        example_prompt=None,
        output_format_prompt=None,
        experiment_name="",
        elitism=True,
        HPO=False,
        mutation_prompts=None,
        adaptive_mutation=False,
        adaptive_prompt=False,
        budget=100,
        eval_timeout=3600,
        max_workers=10,
        parallel_backend="loky",
        log=True,
        minimization=False,
        _random=False,
        niching: Optional[str] = None,
        distance_metric: Optional[Callable[[Solution, Solution], float]] = None,
        niche_radius: Optional[float] = None,
        adaptive_niche_radius: bool = False,
        clearing_interval: Optional[int] = None,
        evaluate_population=False,
        diff_mode: bool = False,
        adaptive_outer_loop: bool = False,
        outer_loop_gamma: float = 0.9,
        outer_loop_tau: float = 3.0,
    ):
        """
        Initializes the LLaMEA instance with provided parameters. Note that by default LLaMEA maximizes the objective.

        Args:
            f (callable): The evaluation function to measure the fitness of algorithms.
            llm (object): An instance of a language model that will be used to generate and evolve algorithms.
            n_parents (int): The number of parents in the population.
            n_offspring (int): The number of offspring each iteration.
            elitism (bool): Flag to decide if elitism (plus strategy) should be used in the evolutionary process or comma strategy.
            role_prompt (str): A prompt that defines the role of the language model in the optimization task.
            task_prompt (str): A prompt describing the task for the language model to generate optimization algorithms.
            example_prompt (str): An example prompt to guide the language model in generating code (or None for default).
            output_format_prompt (str): A prompt that specifies the output format of the language model's response.
            experiment_name (str): The name of the experiment for logging purposes.
            elitism (bool): Flag to decide if elitism should be used in the evolutionary process.
            HPO (bool): Flag to decide if hyper-parameter optimization is part of the evaluation function.
                In case it is, a configuration space should be asked from the LLM as additional output in json format.
            mutation_prompts (list): A list of prompts to specify mutation operators to the LLM model. Each mutation, a random choice from this list is made.
            adaptive_mutation (bool): If set to True, the mutation prompt 'Change X% of the lines of code' will be used in an adaptive control setting.
                This overwrites mutation_prompts.
            adaptive_prompt (bool): If True, the task prompt is optimized before each mutation, allowing it to co-evolve with the individuals.
            budget (int): The number of generations to run the evolutionary algorithm.
            eval_timeout (int): The number of seconds one evaluation can maximum take (to counter infinite loops etc.). Defaults to 1 hour.
            max_workers (int): The maximum number of parallel workers to use for evaluating individuals.
            parallel_backend (str): The backend to use for parallel processing (e.g., 'loky', 'threading').
            log (bool): Flag to switch of the logging of experiments.
            minimization (bool): Whether we minimize or maximize the objective function. Defaults to False.
            _random (bool): Flag to switch to random search (purely for debugging).
            niching (str | None): Niching strategy to use. Supports "sharing" and
                "clearing". If ``None``, niching is disabled.
            distance_metric (callable | None): Function that computes a distance
                between two :class:`Solution` objects. Defaults to a simple AST
                based distance if not supplied.
            niche_radius (float | None): Radius for niche determination when
                using fitness sharing or clearing. If ``None`` a default of ``0.5``
                is used when a niching method is active.
            adaptive_niche_radius (bool): If ``True`` the niche radius adapts to
                the population each generation.
            clearing_interval (int | None): Interval (in generations) at which
                clearing is applied when ``niching`` is set to ``"clearing"``.
            evaluate_population (bool): If True, the evaluation function `f` should
                accept a list with the new population and a list of parents (optionally, to deal with elitism)
                and return a list of solutions that are evaluated, also the parents may receive new fitness values and should be returned.
                So `f` should have the signature
                `f(population, parents=None, logger=None) -> (evaluated_offspring, evaluated_parents)`.
            diff_mode (bool): If ``True``, the LLM is asked to generate unified diff
                patches instead of complete code when evolving solutions.
            adaptive_outer_loop (bool): Enables the DTS-controlled mutation/crossover
                controller with strict 1+1 elitism. When ``True`` the classical
                population loop is bypassed.
            outer_loop_gamma (float): Discount factor for the outer-loop bandit.
            outer_loop_tau (float): Posterior variance clamp for the outer-loop bandit.
        """
        self.llm = llm
        self.model = llm.model
        self.diff_mode = diff_mode
        self.eval_timeout = eval_timeout
        self.f = f  # evaluation function, provides an individual as output.
        self.role_prompt = role_prompt
        self.parallel_backend = parallel_backend
        if role_prompt == "":
            self.role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."
        if task_prompt == "":
            self.task_prompt = """
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code to minimize the function value. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.

IMPORTANT: Structure your algorithm with these 4 REQUIRED methods (exact names):
1. `def parent_selection(self, population)` - Select parent individuals from the population for reproduction. Returns selected parents.
2. `def recombination(self, parents)` - Combine/crossover parents to create offspring. Returns offspring candidates.
3. `def mutation(self, candidate)` - Apply mutation operator to a candidate solution. Returns mutated candidate.
4. `def survivor_selection(self, population, offspring)` - Select survivors from population and offspring for the next generation. Returns the new population.

These 4 methods will be used for algorithm recombination in an evolutionary framework. The `__call__` method should orchestrate these 4 methods in a main optimization loop.

Give an excellent and novel heuristic algorithm to solve this task.
"""
        else:
            self.task_prompt = task_prompt

        if example_prompt == None:
            self.example_prompt = """
An example of such code (a simple evolutionary algorithm), is as follows:
```python
import numpy as np

class SimpleEA:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.f_opt = np.inf
        self.x_opt = None
        self.lb = -5.0
        self.ub = 5.0

    def parent_selection(self, population):
        # Tournament selection
        selected = []
        for _ in range(len(population) // 2):
            i, j = np.random.choice(len(population), 2, replace=False)
            if population[i][1] < population[j][1]:
                selected.append(population[i])
            else:
                selected.append(population[j])
        return selected

    def recombination(self, parents):
        # Uniform crossover
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            p1, p2 = parents[i][0], parents[(i + 1) % len(parents)][0]
            mask = np.random.random(self.dim) < 0.5
            child = np.where(mask, p1, p2)
            offspring.append(child)
        return offspring

    def mutation(self, candidate):
        # Gaussian mutation
        mutant = candidate + np.random.normal(0, 0.1, self.dim)
        return np.clip(mutant, self.lb, self.ub)

    def survivor_selection(self, population, offspring):
        # Elitist: keep best from combined pool
        combined = population + offspring
        combined.sort(key=lambda x: x[1])
        return combined[:self.pop_size]

    def __call__(self, func):
        # Initialize population
        population = []
        for _ in range(self.pop_size):
            x = np.random.uniform(self.lb, self.ub, self.dim)
            f = func(x)
            population.append((x, f))
            if f < self.f_opt:
                self.f_opt, self.x_opt = f, x

        evals = self.pop_size
        while evals < self.budget:
            parents = self.parent_selection(population)
            offspring_candidates = self.recombination(parents)
            
            offspring = []
            for candidate in offspring_candidates:
                mutant = self.mutation(candidate)
                f = func(mutant)
                evals += 1
                offspring.append((mutant, f))
                if f < self.f_opt:
                    self.f_opt, self.x_opt = f, mutant
                if evals >= self.budget:
                    break

            population = self.survivor_selection(population, offspring)

        return self.f_opt, self.x_opt
```
"""
        else:
            self.example_prompt = example_prompt

        if output_format_prompt is None:
            self.output_format_prompt = """
Provide the Python code and a one-line description with the main idea (without enters). Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```
"""
            if HPO:
                self.output_format_prompt = """
Provide the Python code, a one-line description with the main idea (without enters) and the SMAC3 Configuration space to optimize the code (in Python dictionary format). Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```
Space: <configuration_space>"""
        else:
            self.output_format_prompt = output_format_prompt
        self.diff_output_format_prompt = """
Provide only the unified diff patch for the requested changes. Begin with
`--- original.py` and `+++ updated.py` headers and enclose the patch in a
markdown code block labelled as diff:
# Description: <short-description>
```diff
--- original.py
+++ updated.py
@@
<patch>
```
"""
        self.mutation_prompts = mutation_prompts
        self.adaptive_mutation = adaptive_mutation
        if mutation_prompts == None:
            self.mutation_prompts = [
                "Refine the strategy of the selected solution to improve it.",  # small mutation
                # "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
            ]
        self.budget = budget
        self.n_parents = n_parents
        self.n_offspring = n_offspring
        self.population = []
        self.elitism = elitism
        self.generation = 0
        self.run_history = []
        self.log = log
        self._random = _random
        self.HPO = HPO
        self.minimization = minimization
        self.evaluate_population = evaluate_population
        self.adaptive_prompt = adaptive_prompt
        self.worst_value = -np.inf
        if minimization:
            self.worst_value = np.inf
        self.niching = niching
        self.distance_metric = distance_metric or code_distance
        self.niche_radius = niche_radius if niche_radius is not None else 0.5
        self.adaptive_niche_radius = adaptive_niche_radius
        self.clearing_interval = clearing_interval
        self.best_so_far = Solution(name="", code="")
        self.best_so_far.set_scores(self.worst_value, "", "")
        self.experiment_name = experiment_name

        if self.log:
            modelname = self.model.replace(":", "_")
            self.logger = ExperimentLogger(f"LLaMEA-{modelname}-{experiment_name}")
            self.llm.set_logger(self.logger)
        else:
            self.logger = None
        self.textlog = logging.getLogger(__name__)
        if max_workers > self.n_offspring:
            max_workers = self.n_offspring
        self.max_workers = max_workers
        self.adaptive_outer_loop = adaptive_outer_loop
        self.outer_loop_gamma = outer_loop_gamma
        self.outer_loop_tau = outer_loop_tau
        
        # ------------------------------------------------------------------ #
        # Two-level bandit hierarchy for MADA 2.0
        # ------------------------------------------------------------------ #
        # Level 1: Outer-loop bandit (Mutation vs Crossover) - DTS
        self.outer_loop_arms = ("mutation", "crossover")
        self.outer_bandit: Optional[DiscountedThompsonSampling] = None
        
        # Level 2: Block-level bandits for MADA Innovation crossover
        # Each block has a 3-arm bandit: alpha, beta, innovation
        # These are managed by DiscountedThompsonSampler (block-aware)
        self.block_bandit_arms = ["alpha", "beta", "innovation"]
        self.block_bandit: Optional[DiscountedThompsonSampler] = None
        
        # Note: Inner loop (mutation type, crossover type) is always 50/50 random
        
        self.block_parser: Optional[BlockParser] = None
        if self.adaptive_outer_loop:
            self._ensure_outer_controller_dependencies()

    def logevent(self, event):
        self.textlog.info(event)

    def initialize_single(self):
        """
        Initializes a single solution.
        """
        new_individual = Solution(name="", code="", generation=self.generation)
        session_messages = [
            {
                "role": "user",
                "content": self.role_prompt
                + self.task_prompt
                + self.example_prompt
                + self.output_format_prompt,
            },
        ]
        try:
            new_individual = self.llm.sample_solution(session_messages, HPO=self.HPO)
            new_individual.generation = self.generation
            new_individual.task_prompt = self.task_prompt
            if not self.evaluate_population:
                new_individual = self.evaluate_fitness(new_individual)
        except Exception as e:
            new_individual.set_scores(
                self.worst_value,
                f"An exception occured: {traceback.format_exc()}.",
                repr(e) + traceback.format_exc(),
            )
            self.logevent(f"An exception occured: {traceback.format_exc()}.")
            if hasattr(self.f, "log_individual"):
                self.f.log_individual(new_individual)

        return new_individual

    def initialize(self):
        """
        Initializes the evolutionary process by generating the first parent population.
        """

        population = []
        # Generate each individual sequentially (one by one)
        for _ in range(self.n_parents):
            try:
                p = self.initialize_single()
                population.append(p)
            except Exception as e:
                print(f"Error initializing individual: {e}")

        if self.evaluate_population:
            population = self.evaluate_population_fitness(population)

        for p in population:
            self.run_history.append(p)

        self.generation += 1
        self.population = population  # Save the entire population
        self.update_best()

    def evaluate_fitness(self, individual):
        """
        Evaluates the fitness of the provided individual by invoking the evaluation function `f`.
        This method handles error reporting and logs the feedback, fitness, and errors encountered.

        Args:
            individual (Solution): The solution instance to evaluate.

        Returns:
            Solution: The updated solution with feedback, fitness and error information filled in.
        """
        with contextlib.redirect_stdout(None):
            updated_individual = self.f(individual, self.logger)

        return updated_individual

    def evaluate_population_fitness(self, new_population):
        """Evaluate a full population of solutions."""
        with contextlib.redirect_stdout(None):
            # pass the new population and the parent population to the evaluation function
            evaluated_offspring, evaluated_parents = self.f(
                new_population, self.population, self.logger
            )
            self.population = evaluated_parents  # The parent population fitness might also be updated (this does not need to be logged)
        return evaluated_offspring

    def optimize_task_prompt(self, individual):
        """Use the LLM to improve the task prompt for a given individual."""
        prompt = f"""{self.role_prompt}
You are tasked with refining the instructions (task prompt) that guides an LLM to generate algorithms.
### Current task prompt:
----
{individual.task_prompt}
----

### The current algorithm generated with that prompt:
```python
{individual.code}
```

### Feedback from the evaluation on this algorithm:
----
{individual.feedback}
----

Provide an improved / rephrased / augmented task prompt only. The intent of the task prompt should stay the same.
"""
        session_messages = [{"role": "user", "content": prompt}]
        try:
            new_prompt = self.llm.query(session_messages)
            return new_prompt.strip()
        except Exception as e:
            self.logevent(f"Prompt optimization failed: {e}")
            return individual.task_prompt

    def construct_prompt(self, individual):
        """
        Constructs a new session prompt for the language model based on a selected individual.

        Args:
            individual (dict): The individual to mutate.

        Returns:
            list: A list of dictionaries simulating a conversation with the language model for the next evolutionary step.
        """
        # Generate the current population summary
        population_summary = "\n".join([ind.get_summary() for ind in self.population])
        solution = individual.code
        description = individual.description
        feedback = individual.feedback
        if self.adaptive_mutation == True:
            num_lines = len(solution.split("\n"))
            prob = discrete_power_law_distribution(num_lines, 1.5)
            new_mutation_prompt = f"""Refine the strategy of the selected solution to improve it. 
Make sure you only change {(prob*100):.1f}% of the code, which means if the code has 100 lines, you can only change {prob*100} lines, and the rest of the lines should remain unchanged. 
This input code has {num_lines} lines, so you can only change {max(1, int(prob*num_lines))} lines, the rest {num_lines-max(1, int(prob*num_lines))} lines should remain unchanged. 
This changing rate {(prob*100):.1f}% is a mandatory requirement, you cannot change more or less than this rate.
"""
            self.mutation_prompts = [new_mutation_prompt]

        mutation_operator = random.choice(self.mutation_prompts)
        individual.set_operator(mutation_operator)

        task_prompt = (
            individual.task_prompt if self.adaptive_prompt else self.task_prompt
        )
        final_prompt = f"""{task_prompt}
The current population of algorithms already evaluated (name, description, score) is:
{population_summary}

The selected solution to update is:
{description}

With code:
{solution}

{feedback}

{mutation_operator}
{self.diff_output_format_prompt if self.diff_mode else self.output_format_prompt}
"""
        session_messages = [
            {"role": "user", "content": self.role_prompt + final_prompt},
        ]

        if self._random:  # not advised to use, only for debugging purposes
            session_messages = [
                {"role": "user", "content": self.role_prompt + self.task_prompt},
            ]
        # Logic to construct the new prompt based on current evolutionary state.
        return session_messages

    def update_best(self):
        """
        Update the best individual in the new population
        """
        if self.minimization == False:
            best_individual = max(self.population, key=lambda x: x.fitness)

            if best_individual.fitness > self.best_so_far.fitness:
                self.best_so_far = best_individual
        else:
            best_individual = min(self.population, key=lambda x: x.fitness)

            if best_individual.fitness < self.best_so_far.fitness:
                self.best_so_far = best_individual

    def adapt_niche_radius(self, population):
        """Adapt the niche radius based on the current population."""
        if not self.adaptive_niche_radius or len(population) < 2:
            return
        dists = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dists.append(self.distance_metric(population[i], population[j]))
        if dists:
            self.niche_radius = float(np.mean(dists))

    def apply_niching(self, population):
        """Apply the configured niching strategy to ``population``."""
        if self.niching not in {"sharing", "clearing"}:
            return population

        self.adapt_niche_radius(population)

        if self.niching == "sharing":
            for i, ind in enumerate(population):
                niche_count = 1.0
                for j, other in enumerate(population):
                    if i == j:
                        continue
                    d = self.distance_metric(ind, other)
                    if d < self.niche_radius and self.niche_radius > 0:
                        niche_count += 1 - d / self.niche_radius
                if self.minimization:
                    ind.fitness *= niche_count
                else:
                    ind.fitness /= niche_count
        elif self.niching == "clearing":
            if self.clearing_interval and self.generation % self.clearing_interval != 0:
                return population
            reverse = self.minimization == False
            population.sort(key=lambda x: x.fitness, reverse=reverse)
            niches = []
            for ind in population:
                if all(
                    self.distance_metric(ind, winner) >= self.niche_radius
                    for winner in niches
                ):
                    niches.append(ind)
                else:
                    ind.fitness = self.worst_value
        return population

    def selection(self, parents, offspring):
        """
        Select the new population based on the parents and the offspring and the current strategy.

        Args:
            parents (list): List of solutions.
            offspring (list): List of new solutions.

        Returns:
            list: List of new selected population.
        """
        reverse = self.minimization == False
        if self.elitism:
            combined_population = parents + offspring
            combined_population = self.apply_niching(combined_population)
            combined_population.sort(key=lambda x: x.fitness, reverse=reverse)
            new_population = combined_population[: self.n_parents]
        else:
            offspring = self.apply_niching(list(offspring))
            offspring.sort(key=lambda x: x.fitness, reverse=reverse)
            new_population = offspring[: self.n_parents]

        return new_population

    def evolve_solution(self, individual):
        """
        Evolves a single solution by constructing a new prompt,
        querying the LLM, and evaluating the fitness.
        """
        individual_copy = individual.copy()
        if self.adaptive_prompt:
            individual_copy.task_prompt = self.optimize_task_prompt(individual_copy)
        new_prompt = self.construct_prompt(individual_copy)

        evolved_individual = individual.empty_copy()
        try:
            evolved_individual = self.llm.sample_solution(
                new_prompt,
                evolved_individual.parent_ids,
                HPO=self.HPO,
                base_code=individual.code,
                diff_mode=self.diff_mode,
            )
            evolved_individual.generation = self.generation
            evolved_individual.task_prompt = individual_copy.task_prompt
            if not self.evaluate_population:
                evolved_individual = self.evaluate_fitness(evolved_individual)
        except Exception as e:
            error = repr(e)
            evolved_individual.generation = self.generation
            evolved_individual.set_scores(
                self.worst_value, f"An exception occurred: {error}.", error
            )
            if hasattr(self.f, "log_individual"):
                self.f.log_individual(evolved_individual)
            self.logevent(f"An exception occured: {traceback.format_exc()}.")

        # self.progress_bar.update(1)
        return evolved_individual

    # ------------------------------------------------------------------ #
    # Adaptive outer loop helpers
    # ------------------------------------------------------------------ #
    def _ensure_outer_controller_dependencies(self):
        """Initialise the bandits and parser needed for the adaptive controller."""

        # Outer-loop bandit: Mutation vs Crossover
        if self.outer_bandit is None:
            self.outer_bandit = DiscountedThompsonSampling(
                n_arms=len(self.outer_loop_arms),
                gamma=self.outer_loop_gamma,
                tau_max=self.outer_loop_tau,
            )
        
        # Block-level bandit for MADA Innovation crossover
        # Each block (parent_selection, recombination, mutation, survivor_selection)
        # has 3 arms: alpha, beta, innovation
        if self.block_bandit is None:
            self.block_bandit = DiscountedThompsonSampler(
                arm_names=self.block_bandit_arms,
                discount=self.outer_loop_gamma,
                tau_max=self.outer_loop_tau,
            )
        
        if self.block_parser is None:
            self.block_parser = BlockParser()

    def _run_adaptive_outer_loop(self):
        """
        Execute the DTS-guided evolutionary process.
        
        Supports both:
        - Strict 1+1 elitism (n_parents=1, n_offspring=1): offspring replaces parent only if better
        - Population-based (μ+λ or μ,λ): generate n_offspring per generation, select best n_parents
        
        The DTS bandit selects between mutation and crossover for each offspring.
        """

        if self.evaluate_population:
            raise ValueError(
                "Adaptive outer loop is incompatible with evaluate_population=True."
            )

        self._ensure_outer_controller_dependencies()
        self.logevent("Initializing first population")
        self.initialize()
        if self.log:
            self.logger.log_population(self.population)
        self._sort_population()

        # Determine if we're in strict 1+1 mode or population mode
        strict_one_plus_one = (self.n_parents == 1 and self.n_offspring == 1)
        
        self.logevent(
            f"Adaptive outer loop: mode={'1+1' if strict_one_plus_one else f'({self.n_parents}+{self.n_offspring})'}"
        )

        while len(self.run_history) < self.budget:
            parents = list(self.population)
            if not parents:
                raise RuntimeError("Adaptive outer loop requires at least one parent.")

            if strict_one_plus_one:
                # --------------------------------------------------------
                # Strict 1+1 elitism: single offspring, replace only if better
                # --------------------------------------------------------
                primary_parent = parents[0]
                secondary_parent = parents[1] if len(parents) > 1 else parents[0]
                arm_index, arm_label = self._select_outer_arm()

                try:
                    if arm_label == "mutation":
                        offspring, diag = self._generate_mutation_offspring(primary_parent)
                    else:
                        offspring, diag = self._generate_crossover_offspring(
                            primary_parent, secondary_parent
                        )
                except Exception as exc:
                    self.logevent(
                        f"Adaptive outer loop fallback due to generation error: {exc}"
                    )
                    offspring, diag = self._generate_mutation_offspring(primary_parent)

                evaluated = diag.pop("evaluated", False)
                offspring.generation = self.generation
                if not evaluated:
                    offspring = self.evaluate_fitness(offspring)

                self.run_history.append(offspring)
                reward = self._compute_reward(primary_parent, offspring)
                self._update_outer_bandit_state(arm_index, reward)
                self._log_outer_loop_event(arm_label, diag, offspring, reward)

                self.generation += 1
                if self.log:
                    self.logger.log_population([offspring])

                # Strict 1+1: replace only if offspring is better
                if reward > 0:
                    self.population[0] = offspring
                self._sort_population()
                self.update_best()

            else:
                # --------------------------------------------------------
                # Population-based: generate n_offspring, then select best n_parents
                # --------------------------------------------------------
                new_offspring = []
                offspring_count = 0
                
                while offspring_count < self.n_offspring and len(self.run_history) < self.budget:
                    # Select parent(s) for this offspring
                    primary_parent = random.choice(parents)
                    secondary_parent = random.choice(parents)
                    
                    # DTS selects mutation or crossover
                    arm_index, arm_label = self._select_outer_arm()

                    try:
                        if arm_label == "mutation":
                            offspring, diag = self._generate_mutation_offspring(primary_parent)
                        else:
                            offspring, diag = self._generate_crossover_offspring(
                                primary_parent, secondary_parent
                            )
                    except Exception as exc:
                        self.logevent(
                            f"Adaptive outer loop fallback due to generation error: {exc}"
                        )
                        offspring, diag = self._generate_mutation_offspring(primary_parent)

                    evaluated = diag.pop("evaluated", False)
                    offspring.generation = self.generation
                    if not evaluated:
                        offspring = self.evaluate_fitness(offspring)

                    self.run_history.append(offspring)
                    
                    # Compute reward relative to the parent used
                    reward = self._compute_reward(primary_parent, offspring)
                    self._update_outer_bandit_state(arm_index, reward)
                    self._log_outer_loop_event(arm_label, diag, offspring, reward)
                    
                    new_offspring.append(offspring)
                    offspring_count += 1

                self.generation += 1
                if self.log:
                    self.logger.log_population(new_offspring)

                # Selection: (μ+λ) or (μ,λ) based on elitism flag
                self.population = self.selection(parents, new_offspring)
                self._sort_population()
                self.update_best()

            self.logevent(
                f"Generation {self.generation}, best so far: {self.best_so_far.fitness}"
            )

        return self.best_so_far

    def _select_outer_arm(self) -> Tuple[int, str]:
        """Draw an arm index/name pair with a random fallback."""

        self._ensure_outer_controller_dependencies()
        try:
            idx = self.outer_bandit.select_arm()
        except Exception:
            idx = random.randrange(len(self.outer_loop_arms))
        return idx, self._outer_arm_label(idx)

    def _outer_arm_label(self, index: int) -> str:
        return self.outer_loop_arms[index % len(self.outer_loop_arms)]

    def _update_outer_bandit_state(self, arm_index: int, reward: float) -> None:
        if not self.outer_bandit:
            return
        try:
            self.outer_bandit.update(arm_index, reward)
        except Exception as exc:
            self.logevent(f"Failed to update outer-loop bandit: {exc}")

    def _update_block_bandits(self, diag: dict, reward: float) -> None:
        """Update block-level bandits after MADA innovation crossover."""
        if not self.block_bandit:
            return
        
        block_selections = diag.get("block_selections", {})
        if not block_selections:
            return
        
        for block, sel in block_selections.items():
            arm_name = sel.get("arm")
            if arm_name:
                try:
                    self.block_bandit.update(block, arm_name, reward)
                except Exception as exc:
                    self.logevent(f"Failed to update block bandit for {block}: {exc}")

    def _choose_mutation_prompt(self) -> Tuple[str, str]:
        """Return the label/text pair for the mutation sub-strategy."""

        if random.random() < 0.5:
            return ("explore", MUTATION_EXPLORE)
        return ("simplify", MUTATION_SIMPLIFY)

    def _generate_mutation_offspring(self, parent: Solution) -> Tuple[Solution, dict]:
        """Mutate ``parent`` using the selected behavioural prompt."""

        label, prompt = self._choose_mutation_prompt()
        original_prompts = list(self.mutation_prompts)
        self.mutation_prompts = [prompt]
        try:
            child = self.evolve_solution(parent)
        finally:
            self.mutation_prompts = original_prompts

        child.set_operator(f"mutation_{label}")
        diag = {"mutation_type": label, "evaluated": True}
        return child, diag

    def _generate_crossover_offspring(
        self, alpha: Solution, beta: Solution
    ) -> Tuple[Solution, dict]:
        """Create a crossover offspring via pure stitching or MADA innovation."""

        if random.random() < 0.5:
            return self._pure_crossover_offspring(alpha, beta)
        return self._mada_innovation_offspring(alpha, beta)

    def _pure_crossover_offspring(
        self, alpha: Solution, beta: Solution
    ) -> Tuple[Solution, dict]:
        """Stitch parent blocks without invoking the LLM."""

        self._ensure_outer_controller_dependencies()
        parsed_alpha = self.block_parser.extract(alpha.code or "")
        parsed_beta = self.block_parser.extract(beta.code or "")
        overrides = {}
        donors = []
        for block in self.block_parser.target_blocks:
            use_alpha = random.random() < 0.5
            donor = alpha if use_alpha else beta
            parsed = parsed_alpha if use_alpha else parsed_beta
            overrides[block] = parsed.blocks.get(block, "")
            donors.append({"block": block, "parent_id": donor.id})

        child_name = f"{parsed_alpha.class_name}Cross{self.generation}_{random.randint(0, 9999)}"
        code = self.block_parser.assemble(parsed_alpha, overrides, class_name=child_name)
        child = Solution(
            code=code,
            name=child_name,
            description=f"Pure crossover of {alpha.name} and {beta.name}",
            generation=self.generation,
            parent_ids=[alpha.id, beta.id],
        )
        child.set_operator("crossover_pure")
        diag = {"crossover_type": "pure", "donors": donors, "prompt": CROSSOVER_PURE}
        return child, diag

    def _mada_innovation_offspring(
        self, alpha: Solution, beta: Solution
    ) -> Tuple[Solution, dict]:
        """
        Create a hybrid algorithm using block-level bandits.
        
        For each block (parent_selection, recombination, mutation, survivor_selection),
        a 3-arm bandit selects:
        - alpha: use block from parent alpha
        - beta: use block from parent beta  
        - innovation: ask LLM to generate a new block
        
        The LLM is then asked to assemble these blocks into a coherent algorithm.
        """
        self._ensure_outer_controller_dependencies()
        
        parsed_alpha = self.block_parser.extract(alpha.code or "")
        parsed_beta = self.block_parser.extract(beta.code or "")
        
        # For each block, use bandit to select source
        block_selections: Dict[str, dict] = {}
        block_snippets: Dict[str, str] = {}
        innovation_blocks = []
        
        for block in self.block_parser.target_blocks:
            # Bandit selects: alpha, beta, or innovation
            arm_name, theta, snapshot_id = self.block_bandit.select_arm(block)
            
            if arm_name == "alpha":
                snippet = parsed_alpha.blocks.get(block, "")
                source = alpha.name
            elif arm_name == "beta":
                snippet = parsed_beta.blocks.get(block, "")
                source = beta.name
            else:  # innovation
                snippet = ""  # Will be generated by LLM
                source = "innovation"
                innovation_blocks.append(block)
            
            block_snippets[block] = snippet
            block_selections[block] = {
                "arm": arm_name,
                "theta": theta,
                "snapshot_id": snapshot_id,
                "source": source,
            }
        
        # Build the prompt with selected blocks and innovation requests
        summary = self._build_population_summary(limit=5)
        
        # Show what blocks we're using from each parent
        block_context = []
        for block in self.block_parser.target_blocks:
            sel = block_selections[block]
            if sel["arm"] == "innovation":
                block_context.append(f"- {block}: GENERATE NEW (innovate)")
            else:
                block_context.append(f"- {block}: from {sel['source']} ({sel['arm']})")
        
        block_context_str = "\n".join(block_context)
        
        # Build the prompt
        if innovation_blocks:
            innovation_instruction = f"""
For the following blocks, generate NEW innovative implementations:
{', '.join(innovation_blocks)}

For other blocks, use the specified parent's implementation."""
        else:
            innovation_instruction = "Use the specified parent implementations for all blocks."
        
        prompt = f"""{self.task_prompt}
Current elite snapshot:
{summary}

Parent Alpha ({alpha.name}):
{alpha.description}
```python
{alpha.code}
```

Parent Beta ({beta.name}):
{beta.description}
```python
{beta.code}
```

Block selection (determined by adaptive bandit):
{block_context_str}

{innovation_instruction}

Assemble a complete algorithm using these block selections. Ensure the code is syntactically correct and all blocks work together coherently.
"""
        session_messages = [
            {
                "role": "user",
                "content": self.role_prompt + prompt + self.output_format_prompt,
            }
        ]
        
        child = self.llm.sample_solution(
            session_messages,
            parent_ids=[alpha.id, beta.id],
            HPO=self.HPO,
            diff_mode=self.diff_mode,
        )
        child.set_operator("crossover_mada")
        
        # Store block selection info for later bandit updates
        diag = {
            "crossover_type": "mada",
            "block_selections": block_selections,
            "innovation_blocks": innovation_blocks,
        }
        return child, diag

    def _build_population_summary(self, limit: int = 5) -> str:
        """Return a short textual summary of the best individuals."""

        reverse = not self.minimization
        population = sorted(self.population, key=lambda x: x.fitness, reverse=reverse)
        summary = [ind.get_summary() for ind in population[:limit]]
        if not summary:
            return "No evaluated individuals yet."
        return "\n".join(summary)

    def _estimate_token_count(self, code: str) -> int:
        """Cheap proxy for tracking code growth."""

        return len(code.split())

    def _compute_reward(self, parent: Solution, offspring: Solution) -> float:
        """Reward is the non-negative fitness improvement."""

        parent_fit = parent.fitness
        child_fit = offspring.fitness
        if any(np.isnan(val) for val in (parent_fit, child_fit)):
            return 0.0
        if self.minimization:
            delta = parent_fit - child_fit
        else:
            delta = child_fit - parent_fit
        return float(max(0.0, delta))

    def _log_outer_loop_event(
        self, arm_label: str, diag: dict, offspring: Solution, reward: float
    ) -> None:
        """Attach metadata, update block bandits if needed, and emit a log line."""

        # Update block-level bandits for MADA innovation crossover
        if diag.get("crossover_type") == "mada":
            self._update_block_bandits(diag, reward)

        event = {
            "selected_arm": arm_label,
            "reward": reward,
            "offspring_token_count": self._estimate_token_count(offspring.code or ""),
        }
        event.update(diag or {})
        offspring.add_metadata("outer_loop_event", event)
        
        # Build log message
        log_parts = [f"[AdaptiveOuterLoop] arm={arm_label}"]
        if event.get("mutation_type"):
            log_parts.append(f"mutation={event['mutation_type']}")
        if event.get("crossover_type"):
            log_parts.append(f"crossover={event['crossover_type']}")
            if event.get("innovation_blocks"):
                log_parts.append(f"innovation_blocks={event['innovation_blocks']}")
        log_parts.append(f"tokens={event['offspring_token_count']}")
        log_parts.append(f"reward={reward:.4f}")
        
        self.logevent(" ".join(log_parts))

    def _sort_population(self):
        """Sort internal population for consistent parent selection."""

        reverse = not self.minimization
        self.population.sort(key=lambda x: x.fitness, reverse=reverse)

    def get_bandit_summary(self) -> dict:
        """Return a summary of all bandit states for logging/debugging."""
        summary = {}
        
        # Outer-loop bandit
        if self.outer_bandit:
            snap = self.outer_bandit.snapshot()
            summary["outer_loop"] = {
                arm: snap[i] for i, arm in enumerate(self.outer_loop_arms)
            }
        
        # Block-level bandits
        if self.block_bandit and self.block_parser:
            summary["block_bandits"] = {}
            for block in self.block_parser.target_blocks:
                try:
                    summary["block_bandits"][block] = self.block_bandit.get_state_snapshot(block)
                except Exception:
                    pass
        
        return summary

    def run(self):
        """
        Main loop to evolve the solutions until the evolutionary budget is exhausted.
        The method iteratively refines solutions through interaction with the language model,
        evaluates their fitness, and updates the best solution found.

        Returns:
            tuple: A tuple containing the best solution and its fitness at the end of the evolutionary process.
        """
        if self.adaptive_outer_loop:
            return self._run_adaptive_outer_loop()
        # self.progress_bar = tqdm(total=self.budget)
        self.logevent("Initializing first population")
        self.initialize()  # Initialize a population
        # self.progress_bar.update(self.n_parents)

        if self.log:
            self.logger.log_population(self.population)

        self.logevent(
            f"Started evolutionary loop, best so far: {self.best_so_far.fitness}"
        )
        while len(self.run_history) < self.budget:
            # pick a new offspring population using random sampling
            new_offspring_population = np.random.choice(
                self.population, self.n_offspring, replace=True
            )

            new_population = []
            # Evolve each offspring sequentially (one by one)
            for individual in new_offspring_population:
                try:
                    p = self.evolve_solution(individual)
                    new_population.append(p)
                except Exception as e:
                    print(f"Error evolving individual: {e}")

            if self.evaluate_population:
                new_population = self.evaluate_population_fitness(new_population)

            for p in new_population:
                self.run_history.append(p)

            self.generation += 1

            if self.log:
                self.logger.log_population(new_population)

            # Update population and the best solution
            self.population = self.selection(self.population, new_population)
            self.update_best()
            self.logevent(
                f"Generation {self.generation}, best so far: {self.best_so_far.fitness}"
            )

        return self.best_so_far
