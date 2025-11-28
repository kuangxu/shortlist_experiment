"""
Simulation script for shortlist selection experiments.

This script implements the experimental plan from detailed_exp_plan.md:
- Runs 6 policies: Uniform, TS, A1 (Standard), A1 (SqrtK), TopTwoTS (Standard), TopTwoTS (SqrtK)
- Uses the "hard" instance: (1, 0.9, ..., 0.9, 0, ..., 0) with k=100, m=10, budget=1000
- Collects time-series data for learning curves
- Outputs results to CSV and JSON files

QUICK TESTING: Modify the DEFAULT_CONFIG dictionary below to quickly change parameters
for testing. Command-line arguments will override these defaults.
"""

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

import algorithms
try:
    import analyze_results
    ANALYZE_AVAILABLE = True
except ImportError:
    ANALYZE_AVAILABLE = False

# ============================================================================
# QUICK TESTING CONFIGURATION
# ============================================================================
# Modify these values for quick testing. These defaults can be overridden
# by command-line arguments.
#
# Example for quick testing (faster runs):
#   DEFAULT_CONFIG["replications"] = 10      # Instead of 1000
#   DEFAULT_CONFIG["budget"] = 200            # Instead of 1000
#   DEFAULT_CONFIG["k"] = 20                  # Instead of 100
#
DEFAULT_CONFIG = {
    "k": 100,              # Total number of arms
    "m": 10,               # Shortlist size
    "budget": 1000,        # Total pulls (including warm start)
    "sigma_sq": 1.0,       # Known variance of rewards
    "replications": 2000,  # Number of experiment replications (reduce for quick testing, e.g., 10-100)
    "seed": 123,           # Random seed for reproducibility
    "log_every": 10,       # Log trajectory data every N steps (0 to disable)
    "x": 0.9,              # Difficulty parameter for hard instance (distractor arm mean)
}
# ============================================================================


def generate_hard_instance(k: int, m: int, x: float = 0.9) -> List[float]:
    """
    Generate the "hard" instance mean vector: (1, x, ..., x, 0, ..., 0).
    
    Structure:
    - 1 Best Arm: μ = 1.0 (index 0)
    - (m-1) Distractor Arms: μ = x (indices 1 to m-1)
    - (k-m) Noise Arms: μ = 0.0 (indices m to k-1)
    
    Args:
        k: Total number of arms
        m: Shortlist size
        x: Difficulty parameter (default 0.9)
    
    Returns:
        List of true mean rewards for each arm
    """
    means = [1.0]  # Best arm
    means.extend([x] * (m - 1))  # Distractor arms
    means.extend([0.0] * (k - m))  # Noise arms
    return means


def compute_optimal_beta(k: int) -> float:
    """
    Compute the worst-case parameter β* for A1 and TopTwoTS.
    
    Formula: β* = (1 + sqrt(k-1)) / (1 + 3*sqrt(k-1))
    
    Args:
        k: Total number of arms
    
    Returns:
        Worst-case beta parameter
    """
    sqrt_k_minus_1 = math.sqrt(k - 1)
    numerator = 1 + sqrt_k_minus_1
    denominator = 1 + 3 * sqrt_k_minus_1
    return numerator / denominator


def classify_arms(k: int, m: int) -> Dict[str, List[int]]:
    """
    Classify arms into groups: best, distractors, and noise.
    
    Args:
        k: Total number of arms
        m: Shortlist size
    
    Returns:
        Dictionary mapping group names to lists of arm indices
    """
    return {
        "best": [0],
        "distractors": list(range(1, m)),
        "noise": list(range(m, k)),
    }


def compute_allocation_stats(
    counts: np.ndarray,
    arm_groups: Dict[str, List[int]],
) -> Dict[str, float]:
    """
    Compute average allocation statistics for each arm group.
    
    Args:
        counts: Array of pull counts per arm
        arm_groups: Dictionary mapping group names to arm indices
    
    Returns:
        Dictionary with average pull counts per group
    """
    stats = {}
    for group_name, arm_indices in arm_groups.items():
        if arm_indices:
            stats[group_name] = float(np.mean(counts[arm_indices]))
        else:
            stats[group_name] = 0.0
    return stats


def run_single_replication(
    rng: np.random.Generator,
    policy_name: str,
    k: int,
    m: int,
    budget: int,
    sigma: float,
    beta: float,
    true_means: Sequence[float],
    log_every: int = 10,
) -> Dict:
    """
    Run a single replication of a policy.
    
    Args:
        rng: Random number generator
        policy_name: Name of the policy to run
        k: Number of arms
        m: Shortlist size
        budget: Total budget
        sigma: Standard deviation of noise
        beta: Beta parameter (for A1 and TopTwoTS)
        true_means: True mean rewards
        log_every: Log trajectory every N steps
    
    Returns:
        Dictionary with results from the replication
    """
    if policy_name == "Uniform":
        result = algorithms.run_uniform_allocation(
            rng, k, m, budget, sigma, true_means, log_every=log_every
        )
    elif policy_name == "Thompson Sampling":
        result = algorithms.run_thompson_sampling(
            rng, k, m, budget, sigma, true_means, log_every=log_every
        )
    elif policy_name == "A1 (Standard)":
        result = algorithms.run_a1_sampling(
            rng, k, m, budget, sigma, beta_top=0.5, true_means=true_means, log_every=log_every
        )
    elif policy_name == "A1 (SqrtK)":
        result = algorithms.run_a1_sampling(
            rng, k, m, budget, sigma, beta_top=beta, true_means=true_means, log_every=log_every
        )
    elif policy_name == "TopTwoTS (Standard)":
        result = algorithms.run_toptwo_ts(
            rng, k, m, budget, sigma, beta=0.5, true_means=true_means, log_every=log_every
        )
    elif policy_name == "TopTwoTS (SqrtK)":
        result = algorithms.run_toptwo_ts(
            rng, k, m, budget, sigma, beta=beta, true_means=true_means, log_every=log_every
        )
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
    
    return result


def generate_figure1_learning_curve(trajectory_data: Dict, policies: List[str]) -> None:
    """
    Generate Figure 1: Learning Curve (Success Probability).
    
    X-Axis: Budget consumed (0 to budget)
    Y-Axis: Probability of including Best Arm in Shortlist
    Content: 6 curves (one for each policy)
    """
    plt.figure(figsize=(10, 6))
    
    colors = {
        "Uniform": "#1f77b4",
        "Thompson Sampling": "#ff7f0e",
        "A1 (Standard)": "#2ca02c",
        "A1 (SqrtK)": "#d62728",
        "TopTwoTS (Standard)": "#9467bd",
        "TopTwoTS (SqrtK)": "#8c564b",
    }
    
    for policy_name in policies:
        if policy_name in trajectory_data:
            data = trajectory_data[policy_name]
            steps = data["steps"]
            success_rates = data["success_rate"]
            
            if steps and success_rates:
                plt.plot(
                    steps,
                    success_rates,
                    label=policy_name,
                    color=colors.get(policy_name, "gray"),
                    linewidth=2,
                )
    
    plt.xlabel("Budget Consumed", fontsize=12)
    plt.ylabel("Probability of Including Best Arm in Shortlist", fontsize=12)
    plt.title("Learning Curve: Success Probability Over Time", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("figure1_learning_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✓ Figure 1 saved: figure1_learning_curve.png")


def generate_figure2_allocation_efficiency(summary_data: List[Dict]) -> None:
    """
    Generate Figure 2: Allocation Efficiency.
    
    Type: Grouped Bar Chart
    Groups: Best Arm, Distractor Arms, Noise Arms
    Y-Axis: Average Sample Count
    Content: Bars for ALL 6 policies
    """
    # Show all policies
    policies = [row["policy"] for row in summary_data]
    best_pulls = [row["best_arm_pulls"] for row in summary_data]
    distractor_pulls = [row["distractor_pulls"] for row in summary_data]
    noise_pulls = [row["noise_pulls"] for row in summary_data]
    
    if not policies:
        print("  ⚠ Figure 2: No data available")
        return
    
    x = np.arange(len(policies))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, best_pulls, width, label="Best Arm", color="#2ca02c")
    bars2 = ax.bar(x, distractor_pulls, width, label="Distractor Arms", color="#ff7f0e")
    bars3 = ax.bar(x + width, noise_pulls, width, label="Noise Arms", color="#d62728")
    
    ax.set_xlabel("Policy", fontsize=12)
    ax.set_ylabel("Average Sample Count", fontsize=12)
    ax.set_title("Allocation Efficiency: Sample Distribution by Arm Group (All Policies)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig("figure2_allocation_efficiency.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✓ Figure 2 saved: figure2_allocation_efficiency.png")


def generate_figure3_posterior_evolution(
    k: int,
    m: int,
    budget: int,
    sigma: float,
    true_means: Sequence[float],
    arm_groups: Dict[str, List[int]],
    beta_optimal: float,
    seed: int,
) -> None:
    """
    Generate Figure 3: Posterior Evolution (Single Path).
    
    X-Axis: Time t
    Y-Axis: Posterior probability P(Arm i = Best)
    Content: Traces for the Best Arm and one Distractor Arm
    """
    # Run a single replication of A1 (SqrtK) to get posterior evolution
    rng = np.random.default_rng(seed)
    best_arm = arm_groups["best"][0]
    distractor_arm = arm_groups["distractors"][0] if arm_groups["distractors"] else None
    
    if distractor_arm is None:
        print("  ⚠ Figure 3: No distractor arms available")
        return
    
    # Initialize
    counts, sums, pulls_used = algorithms.warm_start(rng, k, sigma, true_means)
    remaining_budget = max(budget - pulls_used, 0)
    
    # Track posterior probabilities over time
    time_steps = []
    best_arm_probs = []
    distractor_arm_probs = []
    
    # Compute initial posterior probabilities
    posterior_means, posterior_variances = algorithms.posterior_parameters(counts, sums, sigma)
    
    # Approximate P(arm i is best) using Monte Carlo sampling from posterior
    # We'll sample many times and count how often each arm wins
    n_samples = 1000
    
    for step in range(remaining_budget):
        # Compute current posterior
        posterior_means, posterior_variances = algorithms.posterior_parameters(counts, sums, sigma)
        
        # Sample from posterior to estimate P(arm is best)
        samples = rng.normal(
            loc=posterior_means, scale=np.sqrt(posterior_variances), size=(n_samples, k)
        )
        best_in_samples = np.argmax(samples, axis=1)
        
        best_arm_prob = np.mean(best_in_samples == best_arm)
        distractor_arm_prob = np.mean(best_in_samples == distractor_arm)
        
        time_steps.append(pulls_used + step + 1)
        best_arm_probs.append(best_arm_prob)
        distractor_arm_probs.append(distractor_arm_prob)
        
        # A1 algorithm step
        samples1 = rng.normal(loc=posterior_means, scale=np.sqrt(posterior_variances))
        shortlist = np.argsort(samples1)[-m:][::-1]
        shortlist_best = shortlist[0]
        
        # Find challenger
        while True:
            challenger_samples = rng.normal(loc=posterior_means, scale=np.sqrt(posterior_variances))
            challenger = int(np.argmax(challenger_samples))
            if challenger not in shortlist:
                break
        
        # Choose arm
        if rng.uniform() < beta_optimal:
            chosen_arm = int(shortlist_best)
        else:
            chosen_arm = challenger
        
        # Pull and update
        reward = rng.normal(loc=true_means[chosen_arm], scale=sigma)
        counts[chosen_arm] += 1
        sums[chosen_arm] += reward
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, best_arm_probs, label=f"Best Arm (index {best_arm})", linewidth=2, color="#2ca02c")
    plt.plot(
        time_steps,
        distractor_arm_probs,
        label=f"Distractor Arm (index {distractor_arm})",
        linewidth=2,
        color="#ff7f0e",
    )
    
    plt.xlabel("Time $t$", fontsize=12)
    plt.ylabel("Posterior Probability $P(\\text{Arm } i = \\text{Best})$", fontsize=12)
    plt.title("Posterior Evolution: Confidence Accumulation (A1 SqrtK, Single Path)", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("figure3_posterior_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✓ Figure 3 saved: figure3_posterior_evolution.png")


def run_experiments(
    k: int = 100,
    m: int = 10,
    budget: int = 1000,
    sigma_sq: float = 1.0,
    replications: int = 1000,
    seed: int = 123,
    log_every: int = 10,
    x: float = 0.9,
    data_dir: str = "data",
    simulate_only: bool = False,
) -> None:
    """
    Run all experiments according to the plan in detailed_exp_plan.md.
    
    Args:
        k: Total number of arms
        m: Shortlist size
        budget: Total budget (pulls)
        sigma_sq: Noise variance
        replications: Number of replications
        seed: Random seed
        log_every: Log trajectory every N steps
        x: Difficulty parameter for hard instance
        data_dir: Directory to save data files
        simulate_only: If True, skip plotting and only save data
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    sigma = math.sqrt(sigma_sq)
    beta_optimal = compute_optimal_beta(k)
    
    # Generate hard instance
    true_means = generate_hard_instance(k, m, x)
    arm_groups = classify_arms(k, m)
    
    # Define all policies
    policies = [
        "Uniform",
        "Thompson Sampling",
        "A1 (Standard)",
        "A1 (SqrtK)",
        "TopTwoTS (Standard)",
        "TopTwoTS (SqrtK)",
    ]
    
    print(f"Running experiments with configuration:")
    print(f"  k={k}, m={m}, budget={budget}, replications={replications}")
    print(f"  sigma^2={sigma_sq:.3f}, beta*={beta_optimal:.4f}, x={x}")
    print(f"  Best arm: μ=1.0, Distractors: μ={x} (×{m-1}), Noise: μ=0.0 (×{k-m})")
    print()
    
    # Initialize RNG
    rng = np.random.default_rng(seed)
    
    # Storage for results
    all_results = {policy: [] for policy in policies}
    all_trajectories = {policy: [] for policy in policies}  # List of trajectories per policy
    all_allocations = {policy: [] for policy in policies}
    
    # Run all replications for all policies
    for rep in range(replications):
        if (rep + 1) % 100 == 0:
            print(f"Replication {rep + 1}/{replications}...")
        
        for policy_name in policies:
            # Determine beta parameter
            if "SqrtK" in policy_name:
                beta = beta_optimal
            elif "Standard" in policy_name:
                beta = 0.5
            else:
                beta = 0.5  # Not used for Uniform/TS, but needed for function signature
            
            # Run single replication
            result = run_single_replication(
                rng, policy_name, k, m, budget, sigma, beta, true_means, log_every=log_every
            )
            
            # Store results
            all_results[policy_name].append({
                "success": result["success"],
                "regret": result["regret"],
            })
            
            # Store trajectory data from all replications
            all_trajectories[policy_name].append(result["trajectory"])
            
            # Store allocation stats
            allocation_stats = compute_allocation_stats(result["final_counts"], arm_groups)
            all_allocations[policy_name].append(allocation_stats)
    
    print("\nComputing aggregate statistics...")
    
    # Compute aggregate statistics
    summary_data = []
    for policy_name in policies:
        successes = [r["success"] for r in all_results[policy_name]]
        regrets = [r["regret"] for r in all_results[policy_name]]
        
        success_rate = np.mean(successes)
        avg_regret = np.mean(regrets)
        std_regret = np.std(regrets)
        
        # Allocation statistics
        avg_allocations = {}
        for group_name in ["best", "distractors", "noise"]:
            group_counts = [a[group_name] for a in all_allocations[policy_name]]
            avg_allocations[group_name] = float(np.mean(group_counts))
        
        summary_data.append({
            "policy": policy_name,
            "success_rate": success_rate,
            "avg_regret": avg_regret,
            "std_regret": std_regret,
            "best_arm_pulls": avg_allocations["best"],
            "distractor_pulls": avg_allocations["distractors"],
            "noise_pulls": avg_allocations["noise"],
        })
    
    # Save configuration metadata
    config_data = {
        "k": k,
        "m": m,
        "budget": budget,
        "sigma_sq": sigma_sq,
        "replications": replications,
        "seed": seed,
        "log_every": log_every,
        "x": x,
        "beta_optimal": beta_optimal,
        "policies": policies,
    }
    config_path = os.path.join(data_dir, "config.json")
    print(f"Writing {config_path}...")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    
    # Write results_summary.csv
    summary_path = os.path.join(data_dir, "results_summary.csv")
    print(f"Writing {summary_path}...")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "success_rate",
                "avg_regret",
                "std_regret",
                "best_arm_pulls",
                "distractor_pulls",
                "noise_pulls",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_data)
    
    # Write trajectory_data.json
    print("Writing trajectory_data.json...")
    # Aggregate trajectories across replications for learning curves
    trajectory_data = {}
    for policy_name in policies:
        # Get all trajectories for this policy
        policy_trajectories = all_trajectories[policy_name]
        
        # Find the maximum number of time steps (in case trajectories differ slightly)
        max_steps = max(len(traj) for traj in policy_trajectories) if policy_trajectories else 0
        
        if max_steps == 0:
            trajectory_data[policy_name] = {"steps": [], "success_rate": [], "avg_regret": []}
            continue
        
        # Aggregate success rates and regrets at each time step
        steps = []
        success_rates = []
        avg_regrets = []
        
        # Use the first trajectory to get step indices
        if policy_trajectories:
            first_trajectory = policy_trajectories[0]
            for idx in range(len(first_trajectory)):
                step = first_trajectory[idx][0]
                # Collect success/regret values at this step from all replications
                step_successes = []
                step_regrets = []
                
                for traj in policy_trajectories:
                    if idx < len(traj) and traj[idx][0] == step:
                        step_successes.append(traj[idx][1])
                        step_regrets.append(traj[idx][2])
                
                if step_successes:
                    steps.append(step)
                    success_rates.append(np.mean(step_successes))
                    avg_regrets.append(np.mean(step_regrets))
        
        trajectory_data[policy_name] = {
            "steps": steps,
            "success_rate": [float(x) for x in success_rates],
            "avg_regret": [float(x) for x in avg_regrets],
        }
    
    trajectory_path = os.path.join(data_dir, "trajectory_data.json")
    print(f"Writing {trajectory_path}...")
    with open(trajectory_path, "w") as f:
        json.dump(trajectory_data, f, indent=2)
    
    # Write allocation_data.csv
    allocation_path = os.path.join(data_dir, "allocation_data.csv")
    print(f"Writing {allocation_path}...")
    allocation_rows = []
    for policy_name in policies:
        for group_name in ["best", "distractors", "noise"]:
            group_counts = [a[group_name] for a in all_allocations[policy_name]]
            avg_count = float(np.mean(group_counts))
            std_count = float(np.std(group_counts))
            allocation_rows.append({
                "policy": policy_name,
                "arm_group": group_name,
                "avg_pulls": avg_count,
                "std_pulls": std_count,
            })
    
    with open(allocation_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["policy", "arm_group", "avg_pulls", "std_pulls"]
        )
        writer.writeheader()
        writer.writerows(allocation_rows)
    
    print(f"\nResults written to {data_dir}/:")
    print("  - config.json")
    print("  - results_summary.csv")
    print("  - trajectory_data.json")
    print("  - allocation_data.csv")
    
    if not simulate_only:
        print("\nRunning analysis and generating plots...")
        if ANALYZE_AVAILABLE:
            # Use analyze_results.py for consistent plotting and summary
            analyze_results.analyze_results(data_dir=data_dir, output_dir=".", skip_summary=False)
        else:
            # Fallback to old plotting functions if analyze_results is not available
            # Print summary first
            print("\n" + "=" * 80)
            print("SUMMARY")
            print("=" * 80)
            print(f"{'Policy':<25} {'Success Rate':>15} {'Avg Regret':>15} {'Best Pulls':>15}")
            print("-" * 80)
            for row in summary_data:
                print(
                    f"{row['policy']:<25} "
                    f"{row['success_rate']:>14.3%} "
                    f"{row['avg_regret']:>15.4f} "
                    f"{row['best_arm_pulls']:>15.2f}"
                )
            print("=" * 80)
            print("\nGenerating plots...")
            generate_figure1_learning_curve(trajectory_data, policies)
            generate_figure2_allocation_efficiency(summary_data)
            generate_figure3_posterior_evolution(
                k, m, budget, sigma, true_means, arm_groups, beta_optimal, seed + 9999
            )
            print("  - figure1_learning_curve.png")
            print("  - figure2_allocation_efficiency.png")
            print("  - figure3_posterior_evolution.png")
    else:
        # Print summary when in simulate-only mode
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Policy':<25} {'Success Rate':>15} {'Avg Regret':>15} {'Best Pulls':>15}")
        print("-" * 80)
        for row in summary_data:
            print(
                f"{row['policy']:<25} "
                f"{row['success_rate']:>14.3%} "
                f"{row['avg_regret']:>15.4f} "
                f"{row['best_arm_pulls']:>15.2f}"
            )
        print("=" * 80)
        print("\nSimulation-only mode: skipping plot generation.")
        print("Run 'python analyze_results.py' to generate plots from saved data.")


def main() -> None:
    """Main entry point for the experiment script."""
    parser = argparse.ArgumentParser(
        description="Run shortlist selection experiments per detailed_exp_plan.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick Testing: Modify DEFAULT_CONFIG at the top of this file to quickly change
parameters for testing. Command-line arguments override the defaults.

Example for quick testing (fewer replications):
  python run_experiment.py --replications 10
        """,
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_CONFIG["k"], help="Total number of arms"
    )
    parser.add_argument(
        "--m", type=int, default=DEFAULT_CONFIG["m"], help="Shortlist size"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_CONFIG["budget"],
        help="Total pulls (including warm start)",
    )
    parser.add_argument(
        "--sigma-sq",
        type=float,
        default=DEFAULT_CONFIG["sigma_sq"],
        help="Known variance of rewards",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=DEFAULT_CONFIG["replications"],
        help="Number of experiment replications (reduce for quick testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG["seed"],
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_CONFIG["log_every"],
        help="Log trajectory data every N steps (0 to disable)",
    )
    parser.add_argument(
        "--x",
        type=float,
        default=DEFAULT_CONFIG["x"],
        help="Difficulty parameter for hard instance (distractor arm mean)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to save data files (default: data/)",
    )
    parser.add_argument(
        "--simulate-only",
        action="store_true",
        help="Run simulation only, skip plot generation",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.budget < args.k:
        raise ValueError("--budget must be at least the number of arms to accommodate warm start.")
    if args.m <= 0 or args.m > args.k:
        raise ValueError("--m must satisfy 0 < m <= k.")
    if args.replications <= 0:
        raise ValueError("--replications must be positive.")
    
    run_experiments(
        k=args.k,
        m=args.m,
        budget=args.budget,
        sigma_sq=args.sigma_sq,
        replications=args.replications,
        seed=args.seed,
        log_every=args.log_every,
        x=args.x,
        data_dir=args.data_dir,
        simulate_only=args.simulate_only,
    )


if __name__ == "__main__":
    main()

