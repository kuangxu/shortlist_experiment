"""
Analysis and plotting script for shortlist selection experiments.

This script reads simulation data from the data/ folder and generates plots,
summaries, and visualizations.

Prerequisites:
    Run the simulation first using run_experiment.py to generate the required
    data files in the data/ directory.

Required Input Files (in data/ directory):
    - config.json: Experiment configuration (k, m, budget, etc.)
    - results_summary.csv: Summary statistics for each policy
    - trajectory_data.json: Time-series data for learning curves (optional)
    - difficulty_sweep.json: Difficulty sweep results (optional, for Figure 4)

Usage:
    Basic usage (uses default data/ directory and current directory for output):
        python analyze_results.py
    
    Specify custom data directory:
        python analyze_results.py --data-dir /path/to/data
    
    Specify custom output directory for plots:
        python analyze_results.py --data-dir data --output-dir /path/to/output

Command-line Options:
    --data-dir DIR     Directory containing simulation data files (default: data/)
    --output-dir DIR   Directory to save generated plots (default: current directory)

Output Files:
    - figure1_learning_curve.png: Success probability over time for each policy
    - figure2_allocation_efficiency.png: Sample distribution by arm group
    - figure3_posterior_evolution.png: Posterior probability evolution over time
    - figure4_difficulty_sweep.png: Success probability vs. difficulty level (if data available)

The script also prints a summary table to stdout with success rates, regret,
and allocation statistics for each policy.
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


def generate_hard_instance(k: int, m: int, x: float = 0.9) -> List[float]:
    """
    Generate the "hard" instance mean vector: (1, x, ..., x, 0, ..., 0).
    
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


def generate_figure1_learning_curve(trajectory_data: Dict, policies: List[str], output_path: str = "figure1_learning_curve.png") -> None:
    """
    Generate Figure 1: Learning Curve (Success Probability).
    
    X-Axis: Budget consumed (0 to budget)
    Y-Axis: Probability of including Best Arm in Shortlist
    Content: 6 curves (one for each policy)
    """
    plt.figure(figsize=(10, 6))
    
    colors = {
        "A1 (Standard)": "#2ca02c",
        "A1 (SqrtK)": "#d62728",
        "Thompson Sampling": "#ff7f0e",
        "TopTwoTS (Standard)": "#9467bd",
        "TopTwoTS (SqrtK)": "#8c564b",
        "Uniform": "#1f77b4",
    }
    
    # Define desired order: A1, Thompson, TopTwo, then Uniform
    policy_order = [
        "A1 (Standard)",
        "A1 (SqrtK)",
        "Thompson Sampling",
        "TopTwoTS (Standard)",
        "TopTwoTS (SqrtK)",
        "Uniform",
    ]
    
    # Reorder policies according to desired order
    ordered_policies = [p for p in policy_order if p in policies]
    # Add any policies not in the standard list at the end
    for p in policies:
        if p not in ordered_policies:
            ordered_policies.append(p)
    
    for policy_name in ordered_policies:
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
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure 1 saved: {output_path}")


def generate_figure2_allocation_efficiency(summary_data: List[Dict], k: int, m: int, output_path: str = "figure2_allocation_efficiency.png") -> None:
    """
    Generate Figure 2: Allocation Efficiency.
    
    Type: Grouped Bar Chart
    Groups: Best Arm, Distractor Arms, Noise Arms
    Y-Axis: Average Sample Count
    Content: Bars for ALL 6 policies
    """
    if not summary_data:
        print("  ⚠ Figure 2: No data available")
        return
    
    # Define desired order: A1, Thompson, TopTwo, then Uniform
    policy_order = [
        "A1 (Standard)",
        "A1 (SqrtK)",
        "Thompson Sampling",
        "TopTwoTS (Standard)",
        "TopTwoTS (SqrtK)",
        "Uniform",
    ]
    
    # Create a mapping from policy name to index for sorting
    policy_to_index = {policy: i for i, policy in enumerate(policy_order)}
    
    # Sort summary_data according to desired order
    def sort_key(row):
        policy_name = row["policy"]
        return policy_to_index.get(policy_name, 999)  # Unknown policies go to end
    
    sorted_data = sorted(summary_data, key=sort_key)
    
    policies = [row["policy"] for row in sorted_data]
    best_pulls = [row["best_arm_pulls"] for row in sorted_data]
    distractor_pulls = [row["distractor_pulls"] for row in sorted_data]
    noise_pulls = [row["noise_pulls"] for row in sorted_data]
    
    x = np.arange(len(policies))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Determine arm index ranges for legend
    best_arm_idx = "0"
    distractor_arm_indices = f"1-{m-1}" if m > 2 else "1"
    noise_arm_indices = f"{m}-{k-1}" if k - m > 1 else str(m)
    
    bars1 = ax.bar(x - width, best_pulls, width, label=f"Best Arm (index {best_arm_idx})", color="#2ca02c")
    bars2 = ax.bar(x, distractor_pulls, width, label=f"Distractor Arms (indices {distractor_arm_indices})", color="#ff7f0e")
    bars3 = ax.bar(x + width, noise_pulls, width, label=f"Noise Arms (indices {noise_arm_indices})", color="#d62728")
    
    ax.set_xlabel("Policy", fontsize=12)
    ax.set_ylabel("Average Sample Count", fontsize=12)
    ax.set_title("Allocation Efficiency: Sample Distribution by Arm Group (All Policies)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure 2 saved: {output_path}")


def track_posterior_evolution(
    rng: np.random.Generator,
    k: int,
    m: int,
    budget: int,
    sigma: float,
    true_means: Sequence[float],
    best_arm: int,
    distractor_arm: int,
    algorithm_name: str,
    beta_optimal: float,
) -> Dict[str, List[float]]:
    """
    Run a single algorithm and track posterior probabilities over time.
    
    Returns:
        Dictionary with 'time_steps', 'best_arm_probs', 'distractor_arm_probs'
    """
    # Initialize
    counts, sums, pulls_used = algorithms.warm_start(rng, k, sigma, true_means)
    remaining_budget = max(budget - pulls_used, 0)
    
    # Track posterior probabilities over time
    time_steps = []
    best_arm_probs = []
    distractor_arm_probs = []
    
    # Approximate P(arm i is best) using Monte Carlo sampling from posterior
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
        
        # Run algorithm step based on algorithm_name
        if algorithm_name == "A1 (SqrtK)":
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
                
        elif algorithm_name == "Thompson Sampling":
            # Thompson sampling: sample from posterior, pick best
            samples = rng.normal(loc=posterior_means, scale=np.sqrt(posterior_variances))
            chosen_arm = int(np.argmax(samples))
            
        elif algorithm_name == "TopTwoTS (SqrtK)":
            # TopTwoTS algorithm step
            samples1 = rng.normal(loc=posterior_means, scale=np.sqrt(posterior_variances))
            i1 = int(np.argmax(samples1))
            
            # Resample until we find I^(2) != I^(1)
            while True:
                samples2 = rng.normal(loc=posterior_means, scale=np.sqrt(posterior_variances))
                i2 = int(np.argmax(samples2))
                if i2 != i1:
                    break
            
            # Choose between I^(1) and I^(2) based on beta
            if rng.uniform() < beta_optimal:
                chosen_arm = i1
            else:
                chosen_arm = i2
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        # Pull and update
        reward = rng.normal(loc=true_means[chosen_arm], scale=sigma)
        counts[chosen_arm] += 1
        sums[chosen_arm] += reward
    
    return {
        "time_steps": time_steps,
        "best_arm_probs": best_arm_probs,
        "distractor_arm_probs": distractor_arm_probs,
    }


def generate_figure3_posterior_evolution(
    k: int,
    m: int,
    budget: int,
    sigma: float,
    true_means: Sequence[float],
    arm_groups: Dict[str, List[int]],
    beta_optimal: float,
    seed: int,
    output_path: str = "figure3_posterior_evolution.png",
) -> None:
    """
    Generate Figure 3: Posterior Evolution (Single Path).
    
    X-Axis: Time t
    Y-Axis: Posterior probability P(Arm i = Best)
    Content: Traces for the Best Arm and one Distractor Arm for A1 (SqrtK), Thompson Sampling, and TopTwoTS (SqrtK)
    """
    best_arm = arm_groups["best"][0]
    distractor_arm = arm_groups["distractors"][0] if arm_groups["distractors"] else None
    
    if distractor_arm is None:
        print("  ⚠ Figure 3: No distractor arms available")
        return
    
    # Run all three algorithms with different seeds to get independent runs
    algorithms_to_run = ["A1 (SqrtK)", "Thompson Sampling", "TopTwoTS (SqrtK)"]
    algorithm_colors = {
        "A1 (SqrtK)": {"best": "#2ca02c", "distractor": "#7fbf7f"},
        "Thompson Sampling": {"best": "#ff7f0e", "distractor": "#ffb347"},
        "TopTwoTS (SqrtK)": {"best": "#9467bd", "distractor": "#c5a3e0"},
    }
    
    results = {}
    for i, alg_name in enumerate(algorithms_to_run):
        rng = np.random.default_rng(seed + i * 1000)  # Different seed for each algorithm
        results[alg_name] = track_posterior_evolution(
            rng, k, m, budget, sigma, true_means, best_arm, distractor_arm, alg_name, beta_optimal
        )
    
    # Plot
    plt.figure(figsize=(12, 7))
    
    # Plot best arm probabilities for each algorithm
    for alg_name in algorithms_to_run:
        data = results[alg_name]
        plt.plot(
            data["time_steps"],
            data["best_arm_probs"],
            label=f"{alg_name} - Best Arm (index {best_arm})",
            linewidth=2,
            color=algorithm_colors[alg_name]["best"],
            linestyle="-",
        )
    
    # Plot distractor arm probabilities for each algorithm
    for alg_name in algorithms_to_run:
        data = results[alg_name]
    plt.plot(
            data["time_steps"],
            data["distractor_arm_probs"],
            label=f"{alg_name} - Distractor Arm (index {distractor_arm})",
        linewidth=2,
            color=algorithm_colors[alg_name]["distractor"],
            linestyle="--",
    )
    
    # Find maximum posterior probability across all algorithms and arm types
    max_prob = 0.0
    for alg_name in algorithms_to_run:
        data = results[alg_name]
        max_prob = max(max_prob, max(data["best_arm_probs"]), max(data["distractor_arm_probs"]))
    
    # Set y-axis limit to 1.5x the maximum, capped at 1.0 (since these are probabilities)
    y_max = min(1.5 * max_prob, 1.0)
    
    plt.xlabel("Time $t$", fontsize=12)
    plt.ylabel("Posterior Probability $P(\\text{Arm } i = \\text{Best})$", fontsize=12)
    plt.title("Posterior Evolution: Confidence Accumulation (Single Path)", fontsize=14, fontweight="bold")
    plt.legend(fontsize=9, loc="best", ncol=2)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, y_max)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure 3 saved: {output_path}")


def generate_figure4_difficulty_sweep(
    data_dir: str,
    output_path: str = "figure4_difficulty_sweep.png",
) -> bool:
    """
    Generate Figure 4: Difficulty Sweep Analysis.
    
    Creates two side-by-side plots:
    1. Failure probability (1 - success probability) vs. distractor level
    2. Simple regret vs. distractor level
    
    for A1 (Standard), TopTwoTS (Standard), and Thompson Sampling.
    
    Args:
        data_dir: Directory containing difficulty_sweep.json
        output_path: Path to save the figure
    
    Returns:
        True if figure was successfully created, False otherwise
    """
    sweep_path = os.path.join(data_dir, "difficulty_sweep.json")
    if not os.path.exists(sweep_path):
        print(f"  ⚠ Figure 4: Difficulty sweep data file '{sweep_path}' not found.")
        print(f"     Run 'python run_experiment.py --run-sweep' or 'python run_experiment.py --sweep-only' to generate the data.")
        return False
    
    # Load sweep data
    with open(sweep_path, "r") as f:
        sweep_data = json.load(f)
    
    x_values = np.array(sweep_data["x_values"])
    results = sweep_data["results"]
    x_min = sweep_data["x_min"]
    x_max = sweep_data["x_max"]
    
    # Algorithms to plot (in desired order)
    algorithms_to_test = [
        "A1 (Standard)",
        "TopTwoTS (Standard)",
        "Thompson Sampling",
    ]
    
    print(f"  Loading difficulty sweep data: {len(x_values)} x values")
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {
        "A1 (Standard)": "#2ca02c",
        "TopTwoTS (Standard)": "#9467bd",
        "Thompson Sampling": "#ff7f0e",
    }
    
    # Collect data for both plots
    all_failure_probs = []
    all_regrets = []
    has_regret_data = False
    
    for alg_name in algorithms_to_test:
        if alg_name in results:
            # Check if it's the new format (dictionary) or old format (list)
            if isinstance(results[alg_name], dict):
                success_rates = results[alg_name]["success_rates"]
                avg_regrets = results[alg_name].get("avg_regrets", None)
                has_regret_data = avg_regrets is not None
            else:
                # Old format: just a list of success rates
                success_rates = results[alg_name]
                avg_regrets = None
                has_regret_data = False
            
            # Compute failure probabilities (1 - success_rate)
            failure_probs = [1.0 - sr for sr in success_rates]
            all_failure_probs.extend(failure_probs)
            
            # Plot failure probability (left plot)
            ax1.plot(
                x_values,
                failure_probs,
                label=alg_name,
                linewidth=2,
                color=colors[alg_name],
                marker="o",
                markersize=4,
            )
            
            # Plot regret (right plot) if available
            if has_regret_data and avg_regrets:
                all_regrets.extend(avg_regrets)
                ax2.plot(
                    x_values,
                    avg_regrets,
                    label=alg_name,
                    linewidth=2,
                    color=colors[alg_name],
                    marker="o",
                    markersize=4,
                )
    
    # Set up left plot: Failure Probability
    if all_failure_probs:
        y_min = min(all_failure_probs)
        y_max = max(all_failure_probs)
        y_range = y_max - y_min
        
        # Add 5% margin on each side, but ensure we don't go below 0 or above 1
        margin = max(y_range * 0.05, 0.02)
        y_lim_min = max(0.0, y_min - margin)
        y_lim_max = min(1.0, y_max + margin)
        
        # If range is very small, ensure we have some visible range
        if y_lim_max - y_lim_min < 0.1:
            center = (y_min + y_max) / 2
            y_lim_min = max(0.0, center - 0.05)
            y_lim_max = min(1.0, center + 0.05)
    else:
        y_lim_min = -0.02
        y_lim_max = 1.02
    
    ax1.set_xlabel("Distractor Arm Mean ($x$)", fontsize=12)
    ax1.set_ylabel("Failure Probability", fontsize=12)
    ax1.set_title("Failure Probability vs. Difficulty Level", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10, loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(x_min - 0.02, x_max + 0.02)
    ax1.set_ylim(y_lim_min, y_lim_max)
    
    # Set up right plot: Simple Regret
    if has_regret_data and all_regrets:
        y_min = min(all_regrets)
        y_max = max(all_regrets)
        y_range = y_max - y_min
        
        # Add 5% margin on each side
        margin = max(y_range * 0.05, 0.01)
        y_lim_min = max(0.0, y_min - margin)
        y_lim_max = y_max + margin
        
        # If range is very small, ensure we have some visible range
        if y_lim_max - y_lim_min < 0.01:
            center = (y_min + y_max) / 2
            y_lim_min = max(0.0, center - 0.005)
            y_lim_max = center + 0.005
    else:
        # If no regret data, show a message
        y_lim_min = 0.0
        y_lim_max = 1.0
        ax2.text(0.5, 0.5, "Regret data not available\n(Run with updated code)", 
                ha="center", va="center", transform=ax2.transAxes, fontsize=12)
    
    ax2.set_xlabel("Distractor Arm Mean ($x$)", fontsize=12)
    ax2.set_ylabel("Simple Regret", fontsize=12)
    ax2.set_title("Simple Regret vs. Difficulty Level", fontsize=14, fontweight="bold")
    if has_regret_data:
        ax2.legend(fontsize=10, loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(x_min - 0.02, x_max + 0.02)
    if has_regret_data:
        ax2.set_ylim(y_lim_min, y_lim_max)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure 4 saved: {output_path}")
    return True


def print_summary(summary_data: List[Dict]) -> None:
    """Print a formatted summary table of results."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Policy':<25} {'Success Rate':>15} {'Avg Regret':>15} {'Best Pulls':>15} {'Distractor Pulls':>18} {'Noise Pulls':>15}")
    print("-" * 110)
    for row in summary_data:
        print(
            f"{row['policy']:<25} "
            f"{row['success_rate']:>14.3%} "
            f"{row['avg_regret']:>15.4f} "
            f"{row['best_arm_pulls']:>15.2f} "
            f"{row['distractor_pulls']:>18.2f} "
            f"{row['noise_pulls']:>15.2f}"
        )
    print("=" * 110)


def analyze_results(data_dir: str = "data", output_dir: str = ".", skip_summary: bool = False, include_sweep: bool = True) -> None:
    """
    Read simulation data and generate plots and summaries.
    
    Args:
        data_dir: Directory containing simulation data files
        output_dir: Directory to save generated plots
        skip_summary: If True, skip printing the summary table (useful when called from run_experiment.py)
        include_sweep: If True, generate Figure 4 (difficulty sweep) - reads from saved data if available
    """
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory '{data_dir}' does not exist. Run simulation first.")
    
    # Read configuration
    config_path = os.path.join(data_dir, "config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"Configuration file '{config_path}' not found. Run simulation first.")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    k = config["k"]
    m = config["m"]
    budget = config["budget"]
    sigma_sq = config["sigma_sq"]
    x = config["x"]
    beta_optimal = config["beta_optimal"]
    policies = config["policies"]
    seed = config["seed"]
    
    sigma = math.sqrt(sigma_sq)
    
    print(f"Loading data from {data_dir}/...")
    print(f"Configuration: k={k}, m={m}, budget={budget}, replications={config['replications']}")
    print()
    
    # Read results summary
    summary_path = os.path.join(data_dir, "results_summary.csv")
    if not os.path.exists(summary_path):
        raise ValueError(f"Results summary file '{summary_path}' not found.")
    
    summary_data = []
    with open(summary_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            summary_data.append({
                "policy": row["policy"],
                "success_rate": float(row["success_rate"]),
                "avg_regret": float(row["avg_regret"]),
                "std_regret": float(row["std_regret"]),
                "best_arm_pulls": float(row["best_arm_pulls"]),
                "distractor_pulls": float(row["distractor_pulls"]),
                "noise_pulls": float(row["noise_pulls"]),
            })
    
    # Read trajectory data
    trajectory_path = os.path.join(data_dir, "trajectory_data.json")
    if not os.path.exists(trajectory_path):
        print(f"  ⚠ Warning: Trajectory data file '{trajectory_path}' not found. Skipping Figure 1.")
        trajectory_data = {}
    else:
        with open(trajectory_path, "r") as f:
            trajectory_data = json.load(f)
    
    # Print summary (unless skipped)
    if not skip_summary:
        print_summary(summary_data)
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Figure 1: Learning Curve
    if trajectory_data:
        figure1_path = os.path.join(output_dir, "figure1_learning_curve.png")
        generate_figure1_learning_curve(trajectory_data, policies, figure1_path)
    
    # Figure 2: Allocation Efficiency (all policies)
    figure2_path = os.path.join(output_dir, "figure2_allocation_efficiency.png")
    generate_figure2_allocation_efficiency(summary_data, k, m, figure2_path)
    
    # Figure 3: Posterior Evolution
    true_means = generate_hard_instance(k, m, x)
    arm_groups = classify_arms(k, m)
    figure3_path = os.path.join(output_dir, "figure3_posterior_evolution.png")
    generate_figure3_posterior_evolution(
        k, m, budget, sigma, true_means, arm_groups, beta_optimal, seed + 9999, figure3_path
    )
    
    # Figure 4: Difficulty Sweep (reads from saved data)
    figure4_created = False
    if include_sweep:
        print()
        figure4_path = os.path.join(output_dir, "figure4_difficulty_sweep.png")
        figure4_created = generate_figure4_difficulty_sweep(data_dir, figure4_path)
    
    print(f"\nResults written to {output_dir}/:")
    print("  - figure1_learning_curve.png")
    print("  - figure2_allocation_efficiency.png")
    print("  - figure3_posterior_evolution.png")
    if figure4_created:
        print("  - figure4_difficulty_sweep.png")


def main() -> None:
    """Main entry point for the analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze and plot results from shortlist selection experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script reads simulation data from the data/ folder and generates plots
and summaries. Run the simulation first using run_experiment.py.
        """,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing simulation data files (default: data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save generated plots (default: current directory)",
    )
    
    args = parser.parse_args()
    
    analyze_results(data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

