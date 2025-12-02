# Implementation Plan: MADA-LLAMEA 2.0 (Adaptive Outer Loop)

## Objective
Implement a hierarchical, adaptive evolutionary cycle for the LLaMEA framework. This involves adding a **Discounted Thompson Sampling (DTS)** bandit to the outer loop to dynamically select between Mutation and Crossover strategies, while enforcing **1+1 Elitism** and utilizing **dual mutation prompts** (Exploration vs. Simplification).

## Critical Guidelines for Cursor
1.  **Non-Destructive:** Do not delete existing logic until the new logic is verified.
2.  **Context-Aware:** Respect the current file structure (e.g., if classes are in `src/`, put new ones there).
3.  **Strict Elitism:** The system must transition to a 1+1 strategy (only replace parent if offspring is better).

---

## Phase 1: Deep Codebase Analysis (Do this first)
@Cursor, before writing code, scan the project and report the following:
1.  **File Structure Mapping:**
    * Where is the main evolutionary loop? (e.g., `main.py`, `optimizer.py`)
    * Where are the prompts stored? (e.g., `prompts.py`, `llm_utils.py`)
    * Where is the `Algorithm` class or data structure defined?
2.  **Variable Identification:**
    * Identify the variable holding the `population`.
    * Identify the function used to calculate `fitness` or `AOCC`.
    * Identify the current function that calls the LLM (e.g., `get_response`, `chat_completion`).
3.  **Logic Check:**
    * Does the current loop support `1+1` replacement, or is it strictly population-based ($\mu + \lambda$)?
    * Are the "building blocks" (selection, mutation, crossover, survivor) already modularized in the code?

**STOP after Phase 1. Wait for user confirmation that the analysis is correct.**

---

## Phase 2: Implementation Specifications

### 2.1. The Outer Loop Bandit (`bandits.py`)
Create a new utility class for Discounted Thompson Sampling (DTS).
* **Location:** `utils/bandits.py` (or match existing `utils` folder).
* **Math:**
    * **Posterior Sampling:** $\theta \sim \mathcal{N}(\hat{\mu}, \tau^2)$
    * **Update Rule (Discounted):** $N_{t+1} = \gamma N_t + 1$
    * **Mean Update:** $\hat{\mu}_{t+1} = \frac{\gamma \tilde{\mu}_t + Reward}{N_{t+1}}$
* **Configuration:**
    * `n_arms = 2` (Arm 0: Mutation, Arm 1: Crossover).
    * `gamma = 0.9` (Discount factor).
    * `tau_max` (Variance clamp) to prevent explosion.

### 2.2. The Prompts (`prompts.py`)
Ensure these distinct prompts exist to support the "Behaviour Space" findings:
1.  **`MUTATION_EXPLORE`:** "Generate a new algorithm that is different from those tried before."
2.  **`MUTATION_SIMPLIFY`:** "Refine and simplify the selected algorithm to improve it."
3.  **`CROSSOVER_PURE`:** "Combine these functional blocks exactly as provided... Do not innovate new logic."

### 2.3. The Decision Logic (The "Controller")
Refactor the generation step in the main loop. Implement this exact flow:

```python
# Pseudo-code logic to implement
bandit = DiscountedThompsonSampling(n_arms=2)

for generation in range(budget):
    # 1. Select Strategy via Bandit
    arm_index = bandit.select_arm() 
    
    # 2. Execute Strategy
    if arm_index == 0: # MUTATION (Exploration)
        # Randomly choose sub-strategy (50/50)
        if random.random() < 0.5:
            prompt = MUTATION_EXPLORE # "Random New"
        else:
            prompt = MUTATION_SIMPLIFY # "Refine & Simplify"
        offspring = llm_generate(parent, prompt)
        
    elif arm_index == 1: # CROSSOVER (Exploitation)
        # Randomly choose sub-strategy (50/50)
        if random.random() < 0.5:
            # PURE RECOMBINATION (Stitching)
            offspring = stitch_parents(parent_a, parent_b)
        else:
            # INNOVATIVE MADA (Inner Loop)
            # Use existing MADA block-swapping logic here
            offspring = mada_innovate(parent_a, parent_b)

    # 3. Evaluate
    fitness_offspring = evaluate(offspring)
    
    # 4. Calculate Reward (Improvement)
    # Ensure non-negative reward for stability
    reward = max(0, fitness_offspring - parent.fitness)
    
    # 5. Update Bandit
    bandit.update(arm_index, reward)
    
    # 6. Strict 1+1 Elitism
    if fitness_offspring > parent.fitness:
        parent = offspring
        # Log: Improvement found via Arm {arm_index}
    else:
        # Log: Discarded offspring
        pass
```

## Phase 3: Integration and Safety Checks
@Cursor, when integrating:

Imports: Ensure bandits.py is imported correctly avoiding circular dependencies.

Fallback: Wrap the Bandit selection in a try/except block. If the Bandit fails (e.g., NaN values), default to random.choice([0, 1]) so the run doesn't crash.

Logging: You must log the following for every generation (crucial for analysis):

selected_arm (0 or 1)

mutation_type (Explore vs Simplify)

offspring_token_count (To track code bloat)

reward_value

## Phase 4: Final Verification
Dry Run: instruct Cursor to run the code (or a mock of it) for 2 generations.

Check: Does the bandit state dictionary update after a reward?

Check: Does the "Simplify" prompt actually appear in the logs?


