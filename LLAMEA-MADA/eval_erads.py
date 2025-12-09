import importlib.util, numpy as np
from pathlib import Path
from ioh import get_problem, logger

from utils import OverBudgetException, aoc_logger, correct_aoc

def load_algo(path: Path, cls_name: str):
    spec = importlib.util.spec_from_file_location("erads_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)

def evaluate(algo_cls, budget=10_000):
    results = []
    for fid in range(1, 25):
        for iid in (1, 2, 3):
            problem = get_problem(fid, iid, 5)
            l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
            problem.attach_logger(l2)
            for rep in range(3):
                np.random.seed(rep)
                algo = algo_cls(budget)
                try:
                    algo(problem)
                except OverBudgetException:
                    pass

                auc = correct_aoc(problem, l2, budget)
                results.append((fid, iid, rep, auc))
                l2.reset(problem)
                problem.reset()
    return results

def summarize(results):
    by_fid = {}
    for fid, iid, rep, best in results:
        by_fid.setdefault(fid, []).append(best)
    print("Best-so-far per fid (mean/min/max over iid×reps):")
    for fid in sorted(by_fid):
        vals = np.array(by_fid[fid])
        print(f"fid {fid:02d}: mean={vals.mean():.4e}, min={vals.min():.4e}, max={vals.max():.4e}")
    overall = np.mean([r[3] for r in results])
    print(f"\nOverall mean best-so-far: {overall:.4e}")

if __name__ == "__main__":
    algo_cls = load_algo(Path("ERADS.py"), "ERADS_QuantumFluxUltraRefined")
    results = evaluate(algo_cls, budget=10_000)
    summarize(results)