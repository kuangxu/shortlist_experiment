import argparse
import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np


@dataclass
class ExperimentConfig:
    """Static parameters governing one set of replications."""

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
    """Outcome summary for one policy."""

    name: str
    success_count: int
    replications: int

    @property
    def success_rate(self) -> float:
        return self.success_count / self.replications if self.replications else math.nan

    @property
    def std_error(self) -> float:
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
    """Pull each arm once so posterior means/variances are well-defined."""
    counts = np.zeros(k, dtype=np.int64)
    sums = np.zeros(k, dtype=np.float64)

    for arm in range(k):
        reward = rng.normal(loc=true_means[arm], scale=sigma)
        counts[arm] += 1
        sums[arm] += reward

    pulls_used = k
    return counts, sums, pulls_used


def _posterior_parameters(counts: np.ndarray, sums: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-arm posterior moments under the uninformative Gaussian prior."""
    means = sums / counts
    variances = (sigma**2) / counts
    return means, variances


def _final_shortlist(counts: np.ndarray, sums: np.ndarray, m: int) -> np.ndarray:
    """Pick the m arms with largest posterior means."""
    posterior_means = sums / counts
    top_indices = np.argsort(posterior_means)[-m:][::-1]
    return top_indices


def run_a1_sampling(
    rng: np.random.Generator,
    config: ExperimentConfig,
    true_means: Sequence[float],
) -> ExperimentResult:
    """Simulate Algorithm 1 (A1) per design.md."""
    best_arm = int(np.argmax(true_means))

    success_count = 0
    for rep in range(config.replications):
        counts, sums, pulls_used = _warm_start(rng, config.k, config.sigma, true_means)

        remaining_budget = max(config.budget - pulls_used, 0)

        for _ in range(remaining_budget):
            posterior_means, posterior_variances = _posterior_parameters(counts, sums, config.sigma)

            samples = rng.normal(loc=posterior_means, scale=np.sqrt(posterior_variances))
            shortlist = np.argsort(samples)[-config.m:][::-1]
            shortlist_best = shortlist[0]

            while True:
                # Resample until we identify a challenger not already in the shortlist.
                challenger_samples = rng.normal(loc=posterior_means, scale=np.sqrt(posterior_variances))
                challenger = int(np.argmax(challenger_samples))
                if challenger not in shortlist:
                    break

            if rng.uniform() < config.beta_top:
                chosen_arm = int(shortlist_best)
            else:
                chosen_arm = challenger

            reward = rng.normal(loc=true_means[chosen_arm], scale=config.sigma)
            counts[chosen_arm] += 1
            sums[chosen_arm] += reward

        shortlist_final = _final_shortlist(counts, sums, config.m)
        if best_arm in shortlist_final:
            success_count += 1

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
    """Baseline: choose arms uniformly at random after the warm start."""
    best_arm = int(np.argmax(true_means))

    success_count = 0
    for rep in range(config.replications):
        counts, sums, pulls_used = _warm_start(rng, config.k, config.sigma, true_means)
        remaining_budget = max(config.budget - pulls_used, 0)

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
    """Baseline: classic Gaussian Thompson sampling."""
    best_arm = int(np.argmax(true_means))

    success_count = 0
    for rep in range(config.replications):
        counts, sums, pulls_used = _warm_start(rng, config.k, config.sigma, true_means)
        remaining_budget = max(config.budget - pulls_used, 0)

        for _ in range(remaining_budget):
            posterior_means, posterior_variances = _posterior_parameters(counts, sums, config.sigma)
            samples = rng.normal(loc=posterior_means, scale=np.sqrt(posterior_variances))
            chosen_arm = int(np.argmax(samples))

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
    if len(true_means) != k:
        raise ValueError(f"Expected {k} true means, received {len(true_means)}.")
    return list(true_means)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _two_proportion_z_test(
    ref: ExperimentResult,
    challenger: ExperimentResult,
) -> Tuple[float, float]:
    if ref.replications == 0 or challenger.replications == 0:
        return math.nan, math.nan
    p1 = ref.success_rate
    p2 = challenger.success_rate
    pooled_successes = ref.success_count + challenger.success_count
    pooled_trials = ref.replications + challenger.replications
    if pooled_trials == 0:
        return math.nan, math.nan
    pooled = pooled_successes / pooled_trials
    denom = math.sqrt(pooled * (1 - pooled) * (1 / ref.replications + 1 / challenger.replications))
    if denom == 0:
        return math.nan, math.nan
    z = (p1 - p2) / denom
    p_value = 2 * (1 - _normal_cdf(abs(z)))
    return z, p_value


def run_experiments(config: ExperimentConfig, true_means: Sequence[float]) -> List[ExperimentResult]:
    rng = np.random.default_rng(config.seed)
    validated_means = _validate_true_means(true_means, config.k)
    print(
        "Running experiments with configuration:\n"
        f"  k={config.k}, m={config.m}, budget={config.budget}, replications={config.replications}\n"
        f"  sigma^2={config.sigma**2:.3f}, beta_top={config.beta_top}, seed={config.seed}\n"
        f"  debug_every={config.debug_every if config.debug_every else 'off'}\n"
        f"  theta={[round(x, 4) for x in validated_means]}"
    )

    runners: List[Tuple[str, Callable[[np.random.Generator, ExperimentConfig, Sequence[float]], ExperimentResult]]] = [
        ("Uniform Allocation", run_uniform_allocation),
        ("Thompson Sampling", run_thompson_sampling),
        ("A1 Sampling", run_a1_sampling),
    ]

    results = []
    for name, runner in runners:
        print(f"\n--- {name} ---")
        result = runner(rng, config, validated_means)
        print(
            f"{name}: success_count={result.success_count} / {result.replications} "
            f"({result.success_rate:.3%})"
        )
        results.append(result)

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
    if k == 10:
        return [0.5, 0.45, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1]
    raise ValueError(
        "No default true means available for k != 10. "
        "Please specify --true-means explicitly."
    )


def parse_true_means(raw: str, k: int) -> List[float]:
    parts = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(parts) != k:
        raise argparse.ArgumentTypeError(f"--true-means must contain exactly {k} values.")
    return parts


def build_config(args: argparse.Namespace) -> ExperimentConfig:
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
        sigma=math.sqrt(args.sigma_sq),
        beta_top=args.beta_top,
        seed=args.seed,
        debug_every=args.debug_every,
    )


def main() -> None:
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

    if args.true_means:
        true_means = parse_true_means(args.true_means, config.k)
    else:
        true_means = default_true_means(config.k)

    run_experiments(config, true_means)


if __name__ == "__main__":
    main()

