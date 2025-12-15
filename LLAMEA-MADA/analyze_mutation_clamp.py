import json

# Load data
with open('exp-12-14_051158-google-gemini-2.5-flash-mada-v2-experiment-evolutionary/mada_offspring.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# Filter mutation attempts
mut = [d for d in data if d['operator'] == 'mutation']
errors = [d for d in mut if d['error']]
non_err = [d for d in mut if not d['error']]

print(f"=== MUTATION OPERATOR ANALYSIS ===\n")
print(f"Total mutation attempts: {len(mut)}")
print(f"Errors: {len(errors)} ({len(errors)/len(mut)*100:.1f}%)")
print(f"Successful: {len(non_err)} ({len(non_err)/len(mut)*100:.1f}%)")

print(f"\n=== ERROR PENALTIES (clamp=1.0) ===")
for d in errors:
    print(f"  Attempt {d['attempt']}: raw={d['raw_reward']:.3f}, norm={d['normalized_reward']:.3f}")
    print(f"    Error: {d['error'][:60]}")

print(f"\n=== SUCCESSFUL MUTATION REWARDS ===")
if non_err:
    raw_rewards = [d['raw_reward'] for d in non_err]
    norm_rewards = [d['normalized_reward'] for d in non_err]
    print(f"  Raw reward: mean={sum(raw_rewards)/len(raw_rewards):.4f}, range=[{min(raw_rewards):.4f}, {max(raw_rewards):.4f}]")
    print(f"  Normalized: mean={sum(norm_rewards)/len(norm_rewards):.4f}, range=[{min(norm_rewards):.4f}, {max(norm_rewards):.4f}]")

# Load bandit snapshots
print(f"\n=== BANDIT EVOLUTION ===")
with open('exp-12-14_051158-google-gemini-2.5-flash-mada-v2-experiment-evolutionary/bandit_snapshots.jsonl', 'r') as f:
    snapshots = [json.loads(line) for line in f]

for snap in snapshots:
    gen = snap['generation']
    mut_stats = snap['arm_stats']['mutation']
    probs = snap['selection_probs']['mutation']
    print(f"Gen {gen}: mu_hat={mut_stats['mu_hat']:.3f}, pulls={mut_stats['pulls']}, prob={probs:.1%}")

# Calculate impact
print(f"\n=== CLAMP IMPACT ANALYSIS ===")
print(f"With clamp=1.0:")
print(f"  - Each error -> raw_reward = -1.0")
print(f"  - After normalization -> -1.7 to -3.1 (devastating!)")
print(f"  - Mutation mu_hat collapsed from -0.08 -> -2.25")
print(f"  - Selection probability dropped from 30% -> 1.3%")

print(f"\n=== RECOMMENDED CLAMP VALUES ===")
for clamp in [0.3, 0.5, 0.7]:
    print(f"\nWith clamp={clamp}:")
    print(f"  - Error penalty: -{clamp} (vs -1.0)")
    print(f"  - Estimated normalized: -{clamp*2.5:.2f} to -{clamp*3.0:.2f} (vs -2.5 to -3.0)")
    print(f"  - Recovery potential: {'HIGH' if clamp <= 0.5 else 'MODERATE'}")








