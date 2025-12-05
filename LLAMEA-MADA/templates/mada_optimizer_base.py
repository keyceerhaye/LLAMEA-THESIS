import numpy as np


class MADAOptimizerTemplate:
    """
    Minimal yet complete optimizer that demonstrates the required
    modular block structure for MADA. Each block can be replaced or
    refined independently by the LLM.
    """

    def __init__(self, budget=10000, dim=5):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.lb = None
        self.ub = None
        self.rng = np.random.default_rng()
        self.population = []
        self.best_fitness = np.inf
        self.best_candidate = None

    def parent_selection(self, population):
        """
        Tournament selection that leverages helper sampling utilities.
        population: list[tuple[np.ndarray, float]]
        """
        normalized = self._normalize_population(population)
        if not normalized:
            return []
        parents = []
        pool_size = len(normalized)
        for _ in range(self.population_size):
            idxs = self._safe_index_sample(pool_size, 3)
            candidates = [normalized[i] for i in idxs]
            best = min(candidates, key=lambda item: item[1])
            parents.append(best)
        return parents

    def recombination(self, parents):
        """
        Create a trial vector by averaging two sampled parents.
        parents: list[tuple[np.ndarray, float]]
        """
        normalized = self._normalize_population(parents)
        if not normalized:
            raise ValueError("recombination expects at least one parent")
        if len(normalized) == 1:
            return normalized[0][0].copy()
        idxs = self._safe_index_sample(len(normalized), 2)
        vectors = [normalized[i][0] for i in idxs]
        return np.mean(vectors, axis=0)

    def mutation(self, candidate):
        """
        Apply Gaussian noise while respecting the search bounds.
        candidate: np.ndarray
        """
        base = np.array(candidate, copy=True)
        step = self.rng.normal(0.0, 0.1, size=base.shape)
        mutated = base + step
        return self._clip_vector(mutated)

    def survivor_selection(self, population, offspring):
        """
        Merge parents and offspring, keep the best `population_size`.
        population / offspring: list[tuple[np.ndarray, float]]
        """
        combined = self._normalize_population(population) + self._normalize_population(
            offspring
        )
        combined.sort(key=lambda item: item[1])
        trimmed = combined[: self.population_size]
        for candidate, fitness in trimmed:
            self._update_best(candidate, fitness)
        return self._ensure_tuple_records(trimmed)

    def _initialize_population(self, func):
        pop = []
        for _ in range(self.population_size):
            x = self.rng.uniform(self.lb, self.ub)
            f = func(x)
            self._update_best(x, f)
            pop.append((x, f))
        return self._ensure_tuple_records(pop)

    def _update_best(self, candidate, fitness):
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_candidate = np.array(candidate, copy=True)

    def _safe_index_sample(self, pool_size, count):
        if pool_size <= 0:
            raise ValueError("Cannot sample from an empty pool.")
        count = max(1, count)
        replace = count > pool_size
        indices = self.rng.choice(pool_size, size=count, replace=replace)
        return np.atleast_1d(indices)

    def _clip_vector(self, vector):
        return np.clip(vector, self.lb, self.ub)

    def _normalize_population(self, records):
        """
        Ensure the provided records are a list of (candidate, fitness) tuples.
        """
        if not records:
            return []
        normalized = []
        for candidate, fitness in self._iter_population(records):
            normalized.append((candidate, fitness))
        return normalized

    def _iter_population(self, records):
        """
        Yield (candidate, fitness) tuples, unpacking dict/list records safely.
        """
        if not records:
            return
        for entry in records:
            if entry is None:
                continue
            if isinstance(entry, tuple) and len(entry) == 2:
                candidate, fitness = entry
            elif isinstance(entry, list) and len(entry) == 2:
                candidate, fitness = entry
            elif isinstance(entry, dict):
                candidate = entry.get("candidate")
                fitness = entry.get("fitness")
            else:
                continue
            if candidate is None or fitness is None:
                continue
            yield np.array(candidate, copy=True), float(fitness)

    def _ensure_tuple_records(self, records):
        """
        Guarantee downstream blocks receive a list of (candidate, fitness) tuples.
        """
        normalized = self._normalize_population(records)
        if not normalized:
            return []
        return [(np.array(candidate, copy=True), float(fitness)) for candidate, fitness in normalized]

    def __call__(self, func):
        if self.dim is None:
            self.dim = len(func.bounds.lb)

        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        self.best_fitness = np.inf
        self.best_candidate = None

        self.population = self._initialize_population(func)
        evaluations = self.population_size

        self.f_opt = self.best_fitness
        self.x_opt = self.best_candidate.copy()

        while evaluations < self.budget:
            parents = self.parent_selection(self.population)
            trial = self.recombination(parents)
            trial = self.mutation(trial)
            fitness = func(trial)
            evaluations += 1
            self._update_best(trial, fitness)

            offspring = [(trial, fitness)]
            self.population = self.survivor_selection(self.population, offspring)

            if self.best_fitness < self.f_opt:
                self.f_opt = self.best_fitness
                self.x_opt = self.best_candidate.copy()

        return self.f_opt, self.x_opt


