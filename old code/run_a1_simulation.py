"""
Multi-armed bandit simulation for shortlist selection.

This module implements Algorithm 1 (A1) sampling, a bandit algorithm that maintains
a shortlist of promising arms and uses Thompson sampling with exploration/exploitation
trade-offs. It compares A1 against baseline methods: uniform allocation and classic
Thompson sampling.

The goal is to identify the top m arms (shortlist) from k total arms using a fixed
budget of pulls, where each arm's reward follows a Gaussian distribution with known
variance but unknown mean.
"""
import argparse
import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np


@dataclass
class ExperimentConfig:
    """
    Static parameters governing one set of replications.
    
    Attributes:
        k: Total number of arms in the bandit problem
        m: Size of the shortlist (number of top arms to identify)
        budget: Total number of arm pulls allowed (including warm start)
        replications: Number of independent experiment runs for statistical significance
        sigma: Standard deviation of reward noise (known, constant across arms)
        beta_top: Probability of selecting the best arm from shortlist vs. challenger
                 (controls exploration/exploitation balance in A1)
        seed: Random seed for reproducibility (None for non-deterministic)
        debug_every: Print debug info every N replications (0 to disable)
    """

    k: int
    m: int
    budget: int
    replications: int
    sigma: float
    beta_top: float
    seed: int | None
    debug_every: int


@dataclass
class ExperimentResult:
    """
    Outcome summary for one policy (algorithm).
    
    Attributes:
        name: Human-readable name of the policy (e.g., "A1 Sampling")
        success_count: Number of replications where the true best arm was in final shortlist
        replications: Total number of replications run
    """

    name: str
    success_count: int
    replications: int

    @property
    def success_rate(self) -> float:
        """Compute the fraction of replications that successfully identified the best arm."""
        return self.success_count / self.replications if self.replications else math.nan

    @property
    def std_error(self) -> float:
        """
        Compute standard error of the success rate using binomial approximation.
        
        For a binomial proportion p with n trials, SE = sqrt(p(1-p)/n).
        """
        if self.replications == 0:
            return math.nan
        rate = self.success_rate
        return math.sqrt(rate * (1 - rate) / self.replications)


def _warm_start(
    rng: np.random.Generator,
    k: int,
    sigma: float,
    true_means: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Pull each arm once to initialize posterior distributions.
    
    This ensures all arms have at least one observation, making posterior means
    and variances well-defined. Without this, we'd have division by zero when
    computing posterior parameters for unobserved arms.
    
    Args:
        rng: Random number generator
        k: Number of arms
        sigma: Standard deviation of reward noise
        true_means: True mean reward for each arm (used for simulation)
    
    Returns:
        counts: Array of pull counts per arm (all ones after warm start)
        sums: Array of cumulative reward sums per arm
        pulls_used: Number of pulls consumed (always k)
    """
    counts = np.zeros(k, dtype=np.int64)
    sums = np.zeros(k, dtype=np.float64)

    # Pull each arm exactly once
    for arm in range(k):
        reward = rng.normal(loc=true_means[arm], scale=sigma)
        counts[arm] += 1
        sums[arm] += reward

    pulls_used = k
    return counts, sums, pulls_used


def _posterior_parameters(counts: np.ndarray, sums: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute posterior mean and variance for each arm under Gaussian model.
    
    Assumes uninformative prior and known noise variance sigma^2. With n_i pulls
    of arm i, the posterior mean is the sample mean (sum_i / count_i) and the
    posterior variance is sigma^2 / n_i (decreases with more observations).
    
    Args:
        counts: Number of pulls per arm
        sums: Cumulative reward sums per arm
        sigma: Known standard deviation of reward noise
    
    Returns:
        means: Posterior mean for each arm (sample mean)
        variances: Posterior variance for each arm (sigma^2 / count)
    """
    means = sums / counts
    variances = (sigma**2) / counts
    return means, variances


def _final_shortlist(counts: np.ndarray, sums: np.ndarray, m: int) -> np.ndarray:
    """
    Select the top m arms based on posterior means.
    
    This is the final shortlist used to evaluate success: we check if the true
    best arm is included in this list.
    
    Args:
        counts: Number of pulls per arm
        sums: Cumulative reward sums per arm
        m: Size of shortlist to return
    
    Returns:
        Array of m arm indices, sorted in descending order of posterior mean
        (best arm first)
    """
    posterior_means = sums / counts
    # argsort gives ascending order, [-m:] takes last m, [::-1] reverses to descending
    top_indices = np.argsort(posterior_means)[-m:][::-1]
    return top_indices


def run_a1_sampling(
    rng: np.random.Generator,
    config: ExperimentConfig,
    true_means: Sequence[float],
) -> ExperimentResult:
    """
    Simulate Algorithm 1 (A1) per design.md.
    
    A1 maintains a shortlist of m promising arms and uses Thompson sampling with
    a twist: at each step, it samples a shortlist and a challenger arm, then
    probabilistically chooses between the best shortlist arm (exploitation) and
    the challenger (exploration).
    
    Algorithm steps per pull:
    1. Sample from posterior to form a shortlist of m arms
    2. Sample again to find a challenger arm not in the shortlist
    3. With probability beta_top, pull the best shortlist arm; otherwise pull challenger
    4. Update posterior with observed reward
    
    Args:
        rng: Random number generator
        config: Experiment configuration parameters
        true_means: True mean rewards for each arm (used for generating rewards)
    
    Returns:
        ExperimentResult with success statistics
    """
    best_arm = int(np.argmax(true_means))

    success_count = 0
    for rep in range(config.replications):
        # Initialize: pull each arm once
        counts, sums, pulls_used = _warm_start(rng, config.k, config.sigma, true_means)

        remaining_budget = max(config.budget - pulls_used, 0)

        # Main loop: use remaining budget to explore/exploit
        for _ in range(remaining_budget):
            # Compute current posterior beliefs about each arm
            posterior_means, posterior_variances = _posterior_parameters(counts, sums, config.sigma)

            # Step 1: Sample from posterior to form shortlist
            # This is Thompson sampling: sample from posterior, pick top m
            samples = rng.normal(loc=posterior_means, scale=np.sqrt(posterior_variances))
            shortlist = np.argsort(samples)[-config.m:][::-1]  # Top m arms from this sample
            shortlist_best = shortlist[0]  # Best arm in this shortlist

            # Step 2: Find a challenger arm (not in shortlist) for exploration
            # Resample until we get an arm outside the shortlist
            while True:
                challenger_samples = rng.normal(loc=posterior_means, scale=np.sqrt(posterior_variances))
                challenger = int(np.argmax(challenger_samples))
                if challenger not in shortlist:
                    break

            # Step 3: Choose between exploitation (shortlist best) and exploration (challenger)
            # beta_top controls the trade-off: higher = more exploitation
            if rng.uniform() < config.beta_top:
                chosen_arm = int(shortlist_best)  # Exploit: use best from shortlist
            else:
                chosen_arm = challenger  # Explore: try the challenger

            # Step 4: Pull chosen arm and update posterior
            reward = rng.normal(loc=true_means[chosen_arm], scale=config.sigma)
            counts[chosen_arm] += 1
            sums[chosen_arm] += reward

        # Evaluate success: check if true best arm is in final shortlist
        shortlist_final = _final_shortlist(counts, sums, config.m)
        if best_arm in shortlist_final:
            success_count += 1

        # Optional debug output
        if config.debug_every and (rep + 1) % config.debug_every == 0:
            posterior_means = sums / counts
            print(
                f"[A1] Rep {rep + 1}/{config.replications}: "
                f"shortlist={shortlist_final.tolist()}, "
                f"posterior_means={[round(x, 3) for x in posterior_means]}"
            )

    return ExperimentResult("A1 Sampling", success_count, config.replications)


def run_uniform_allocation(
    rng: np.random.Generator,
    config: ExperimentConfig,
    true_means: Sequence[float],
) -> ExperimentResult:
    """
    Baseline: choose arms uniformly at random after the warm start.
    
    This is a naive baseline that doesn't use any learning - it just explores
    all arms equally. Useful for comparison to show that intelligent algorithms
    perform better.
    
    Args:
        rng: Random number generator
        config: Experiment configuration parameters
        true_means: True mean rewards for each arm
    
    Returns:
        ExperimentResult with success statistics
    """
    best_arm = int(np.argmax(true_means))

    success_count = 0
    for rep in range(config.replications):
        counts, sums, pulls_used = _warm_start(rng, config.k, config.sigma, true_means)
        remaining_budget = max(config.budget - pulls_used, 0)

        # Random exploration: no learning, just uniform random selection
        for _ in range(remaining_budget):
            chosen_arm = int(rng.integers(low=0, high=config.k))
            reward = rng.normal(loc=true_means[chosen_arm], scale=config.sigma)
            counts[chosen_arm] += 1
            sums[chosen_arm] += reward

        shortlist_final = _final_shortlist(counts, sums, config.m)
        if best_arm in shortlist_final:
            success_count += 1

        if config.debug_every and (rep + 1) % config.debug_every == 0:
            posterior_means = sums / counts
            print(
                f"[Uniform] Rep {rep + 1}/{config.replications}: "
                f"shortlist={shortlist_final.tolist()}, "
                f"posterior_means={[round(x, 3) for x in posterior_means]}"
            )

    return ExperimentResult("Uniform Allocation", success_count, config.replications)


def run_thompson_sampling(
    rng: np.random.Generator,
    config: ExperimentConfig,
    true_means: Sequence[float],
) -> ExperimentResult:
    """
    Baseline: classic Gaussian Thompson sampling.
    
    Thompson sampling is a well-known bandit algorithm that balances exploration
    and exploitation by sampling from the posterior and selecting the arm with
    the highest sample. Unlike A1, it doesn't maintain an explicit shortlist
    during the learning phase - it just picks the best arm from each sample.
    
    Args:
        rng: Random number generator
        config: Experiment configuration parameters
        true_means: True mean rewards for each arm
    
    Returns:
        ExperimentResult with success statistics
    """
    best_arm = int(np.argmax(true_means))

    success_count = 0
    for rep in range(config.replications):
        counts, sums, pulls_used = _warm_start(rng, config.k, config.sigma, true_means)
        remaining_budget = max(config.budget - pulls_used, 0)

        for _ in range(remaining_budget):
            # Thompson sampling: sample from posterior, pick best
            posterior_means, posterior_variances = _posterior_parameters(counts, sums, config.sigma)
            samples = rng.normal(loc=posterior_means, scale=np.sqrt(posterior_variances))
            chosen_arm = int(np.argmax(samples))  # Pick arm with highest sample

            reward = rng.normal(loc=true_means[chosen_arm], scale=config.sigma)
            counts[chosen_arm] += 1
            sums[chosen_arm] += reward

        shortlist_final = _final_shortlist(counts, sums, config.m)
        if best_arm in shortlist_final:
            success_count += 1

        if config.debug_every and (rep + 1) % config.debug_every == 0:
            posterior_means = sums / counts
            print(
                f"[Thompson] Rep {rep + 1}/{config.replications}: "
                f"shortlist={shortlist_final.tolist()}, "
                f"posterior_means={[round(x, 3) for x in posterior_means]}"
            )

    return ExperimentResult("Thompson Sampling", success_count, config.replications)


def _validate_true_means(true_means: Sequence[float], k: int) -> List[float]:
    """
    Validate that the number of true means matches the number of arms.
    
    Args:
        true_means: Array of true mean rewards
        k: Expected number of arms
    
    Returns:
        List of validated means
    
    Raises:
        ValueError: If length mismatch
    """
    if len(true_means) != k:
        raise ValueError(f"Expected {k} true means, received {len(true_means)}.")
    return list(true_means)


def _normal_cdf(z: float) -> float:
    """
    Compute cumulative distribution function of standard normal distribution.
    
    Uses the error function: Phi(z) = 0.5 * (1 + erf(z/sqrt(2)))
    
    Args:
        z: Standardized value (z-score)
    
    Returns:
        Probability that standard normal <= z
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _two_proportion_z_test(
    ref: ExperimentResult,
    challenger: ExperimentResult,
) -> Tuple[float, float]:
    """
    Perform two-proportion z-test to compare success rates of two policies.
    
    Tests H0: p1 = p2 vs H1: p1 != p2 (two-sided test).
    Uses pooled variance estimate under the null hypothesis.
    
    Args:
        ref: Reference policy result (typically A1)
        challenger: Challenger policy result (baseline to compare against)
    
    Returns:
        Tuple of (z-statistic, p-value)
        Returns (nan, nan) if insufficient data or division by zero
    """
    if ref.replications == 0 or challenger.replications == 0:
        return math.nan, math.nan
    p1 = ref.success_rate
    p2 = challenger.success_rate
    pooled_successes = ref.success_count + challenger.success_count
    pooled_trials = ref.replications + challenger.replications
    if pooled_trials == 0:
        return math.nan, math.nan
    # Pooled proportion under null hypothesis
    pooled = pooled_successes / pooled_trials
    # Standard error of difference using pooled variance
    denom = math.sqrt(pooled * (1 - pooled) * (1 / ref.replications + 1 / challenger.replications))
    if denom == 0:
        return math.nan, math.nan
    # Z-statistic: (p1 - p2) / SE
    z = (p1 - p2) / denom
    # Two-sided p-value: P(|Z| >= |z|)
    p_value = 2 * (1 - _normal_cdf(abs(z)))
    return z, p_value


def run_experiments(config: ExperimentConfig, true_means: Sequence[float]) -> List[ExperimentResult]:
    """
    Run all three policies (Uniform, Thompson, A1) and compare results.
    
    This is the main orchestration function that:
    1. Validates configuration and true means
    2. Runs each policy with the same configuration
    3. Prints summary statistics
    4. Performs statistical tests comparing A1 to baselines
    
    Args:
        config: Experiment configuration
        true_means: True mean rewards for each arm
    
    Returns:
        List of ExperimentResult objects, one per policy
    """
    rng = np.random.default_rng(config.seed)
    validated_means = _validate_true_means(true_means, config.k)
    print(
        "Running experiments with configuration:\n"
        f"  k={config.k}, m={config.m}, budget={config.budget}, replications={config.replications}\n"
        f"  sigma^2={config.sigma**2:.3f}, beta_top={config.beta_top}, seed={config.seed}\n"
        f"  debug_every={config.debug_every if config.debug_every else 'off'}\n"
        f"  theta={[round(x, 4) for x in validated_means]}"
    )

    # Define all policies to test
    runners: List[Tuple[str, Callable[[np.random.Generator, ExperimentConfig, Sequence[float]], ExperimentResult]]] = [
        ("Uniform Allocation", run_uniform_allocation),
        ("Thompson Sampling", run_thompson_sampling),
        ("A1 Sampling", run_a1_sampling),
    ]

    # Run each policy
    results = []
    for name, runner in runners:
        print(f"\n--- {name} ---")
        result = runner(rng, config, validated_means)
        print(
            f"{name}: success_count={result.success_count} / {result.replications} "
            f"({result.success_rate:.3%})"
        )
        results.append(result)

    # Print summary table
    print("\nSummary (per policy):")
    header = f"{'Policy':<22} {'Success':>12} {'Rate':>10} {'StdErr':>10}"
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.name:<22} "
            f"{result.success_count:>6}/{result.replications:<5} "
            f"{result.success_rate:>9.3%} "
            f"{result.std_error:>9.3%}"
        )

    # Statistical comparison: test if A1 is significantly different from baselines
    a1_result = next((r for r in results if r.name == "A1 Sampling"), None)
    if a1_result:
        print("\nTwo-sided z-tests vs A1 Sampling:")
        for result in results:
            if result is a1_result:
                continue
            z_stat, p_value = _two_proportion_z_test(a1_result, result)
            print(
                f"  A1 vs {result.name:<18} "
                f"Δ={a1_result.success_rate - result.success_rate:+.3%}, "
                f"z={z_stat: .2f}, p={p_value:.3g}"
            )

    return results


def default_true_means(k: int) -> List[float]:
    """
    Return default true means for the example problem (k=10).
    
    The default setup has:
    - 3 top arms: 0.5, 0.45, 0.4 (best arm is 0.5)
    - 2 medium arms: 0.4, 0.4
    - 5 poor arms: 0.1 each
    
    This creates a clear separation between good and bad arms, making it
    easier to evaluate if algorithms can identify the best arm.
    
    Args:
        k: Number of arms (must be 10 for default)
    
    Returns:
        List of true mean rewards
    
    Raises:
        ValueError: If k != 10
    """
    if k == 10:
        return [0.5, 0.45, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1]
    raise ValueError(
        "No default true means available for k != 10. "
        "Please specify --true-means explicitly."
    )


def parse_true_means(raw: str, k: int) -> List[float]:
    """
    Parse comma-separated string of true means into a list.
    
    Args:
        raw: Comma-separated string like "0.5,0.45,0.4,..."
        k: Expected number of values
    
    Returns:
        List of parsed float values
    
    Raises:
        argparse.ArgumentTypeError: If wrong number of values
    """
    parts = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(parts) != k:
        raise argparse.ArgumentTypeError(f"--true-means must contain exactly {k} values.")
    return parts


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """
    Validate command-line arguments and build ExperimentConfig.
    
    Performs validation checks:
    - Budget must allow for warm start (at least k pulls)
    - beta_top must be in (0, 1) for valid probability
    - Shortlist size m must be between 1 and k
    - Must have at least one replication
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Validated ExperimentConfig object
    
    Raises:
        ValueError: If any validation fails
    """
    if args.budget < args.k:
        raise ValueError("--budget must be at least the number of arms to accommodate the warm start.")
    if not (0.0 < args.beta_top < 1.0):
        raise ValueError("--beta-top must be between 0 and 1.")
    if args.m <= 0 or args.m > args.k:
        raise ValueError("--m must satisfy 0 < m <= k.")
    if args.replications <= 0:
        raise ValueError("--replications must be positive.")

    return ExperimentConfig(
        k=args.k,
        m=args.m,
        budget=args.budget,
        replications=args.replications,
        sigma=math.sqrt(args.sigma_sq),  # Convert variance to standard deviation
        beta_top=args.beta_top,
        seed=args.seed,
        debug_every=args.debug_every,
    )


def main() -> None:
    """
    Main entry point: parse arguments and run experiments.
    
    Sets up command-line interface, validates inputs, and orchestrates
    the comparison of A1 sampling against baseline methods.
    """
    parser = argparse.ArgumentParser(
        description="Simulate A1 sampling, uniform allocation, and Thompson sampling per design.md."
    )
    parser.add_argument("--k", type=int, default=10, help="Total number of arms.")
    parser.add_argument("--m", type=int, default=3, help="Shortlist size.")
    parser.add_argument("--budget", type=int, default=100, help="Total pulls (including warm start).")
    parser.add_argument("--replications", type=int, default=1000, help="Number of experiment replications.")
    parser.add_argument("--sigma-sq", type=float, default=1.0, help="Known variance of rewards.")
    parser.add_argument("--beta-top", type=float, default=0.5, help="Probability of sampling the best shortlist arm.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    parser.add_argument(
        "--debug-every",
        type=int,
        default=0,
        help="Print posterior snapshot every N replications (0 disables).",
    )
    parser.add_argument(
        "--true-means",
        type=str,
        default="",
        help="Comma-separated list of true means for each arm. "
        "If omitted, uses the example from design.md when k=10.",
    )

    args = parser.parse_args()
    config = build_config(args)

    # Parse or use default true means
    if args.true_means:
        true_means = parse_true_means(args.true_means, config.k)
    else:
        true_means = default_true_means(config.k)

    # Run all experiments and print results
    run_experiments(config, true_means)


if __name__ == "__main__":
    main()

