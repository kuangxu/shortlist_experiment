"""
Core bandit algorithms for shortlist selection.

This module contains the core algorithm implementations extracted from the simulation
framework. It provides pure algorithm functions that can be used by different simulation
scripts.

Algorithms implemented:
- Uniform Allocation: Baseline random selection
- Thompson Sampling: Classic Gaussian Thompson sampling
- A1 Sampling: Algorithm 1 with shortlist and challenger mechanism
- TopTwoTS: Top-Two Thompson Sampling
"""

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


def warm_start(
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


def posterior_parameters(counts: np.ndarray, sums: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
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


def final_shortlist(counts: np.ndarray, sums: np.ndarray, m: int) -> np.ndarray:
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


def compute_simple_regret(
    true_means: Sequence[float],
    shortlist: np.ndarray,
    counts: np.ndarray,
    sums: np.ndarray,
) -> float:
    """
    Compute simple regret: difference between best arm's mean and highest mean in shortlist.
    
    If the best arm is in the shortlist, regret is 0.
    Otherwise, regret is the difference between the best arm's mean and the highest
    mean in the selected shortlist.
    
    Args:
        true_means: True mean rewards for each arm
        shortlist: Array of arm indices in the final shortlist
        counts: Number of pulls per arm
        sums: Cumulative reward sums per arm
    
    Returns:
        Simple regret value (non-negative)
    """
    best_arm = int(np.argmax(true_means))
    best_mean = true_means[best_arm]
    
    if best_arm in shortlist:
        return 0.0
    
    # Compute empirical means for shortlist arms
    posterior_means = sums / counts
    shortlist_means = posterior_means[shortlist]
    max_shortlist_mean = float(np.max(shortlist_means))
    
    return max(0.0, best_mean - max_shortlist_mean)


def run_uniform_allocation(
    rng: np.random.Generator,
    k: int,
    m: int,
    budget: int,
    sigma: float,
    true_means: Sequence[float],
    log_every: int = 10,
) -> Dict:
    """
    Baseline: choose arms uniformly at random after the warm start.
    
    This is a naive baseline that doesn't use any learning - it just explores
    all arms equally. Useful for comparison to show that intelligent algorithms
    perform better.
    
    Args:
        rng: Random number generator
        k: Number of arms
        m: Shortlist size
        budget: Total number of pulls (including warm start)
        sigma: Standard deviation of reward noise
        true_means: True mean rewards for each arm
        log_every: Log trajectory data every N steps (0 to disable)
    
    Returns:
        Dictionary with:
        - success: Whether best arm is in final shortlist (bool)
        - regret: Simple regret value (float)
        - final_counts: Array of pull counts per arm
        - trajectory: List of (step, success, regret) tuples if log_every > 0
    """
    best_arm = int(np.argmax(true_means))
    
    # Initialize: pull each arm once
    counts, sums, pulls_used = warm_start(rng, k, sigma, true_means)
    remaining_budget = max(budget - pulls_used, 0)
    
    trajectory = []
    total_pulls = pulls_used
    
    # Random exploration: no learning, just uniform random selection
    for step in range(remaining_budget):
        chosen_arm = int(rng.integers(low=0, high=k))
        reward = rng.normal(loc=true_means[chosen_arm], scale=sigma)
        counts[chosen_arm] += 1
        sums[chosen_arm] += reward
        total_pulls += 1
        
        # Log trajectory at intermediate steps
        if log_every > 0 and (step + 1) % log_every == 0:
            shortlist_current = final_shortlist(counts, sums, m)
            success_current = best_arm in shortlist_current
            regret_current = compute_simple_regret(true_means, shortlist_current, counts, sums)
            trajectory.append((total_pulls, success_current, regret_current))
    
    # Final evaluation
    shortlist_final = final_shortlist(counts, sums, m)
    success = best_arm in shortlist_final
    regret = compute_simple_regret(true_means, shortlist_final, counts, sums)
    
    return {
        "success": success,
        "regret": regret,
        "final_counts": counts.copy(),
        "trajectory": trajectory,
    }


def run_thompson_sampling(
    rng: np.random.Generator,
    k: int,
    m: int,
    budget: int,
    sigma: float,
    true_means: Sequence[float],
    log_every: int = 10,
) -> Dict:
    """
    Baseline: classic Gaussian Thompson sampling.
    
    Thompson sampling is a well-known bandit algorithm that balances exploration
    and exploitation by sampling from the posterior and selecting the arm with
    the highest sample. Unlike A1, it doesn't maintain an explicit shortlist
    during the learning phase - it just picks the best arm from each sample.
    
    Args:
        rng: Random number generator
        k: Number of arms
        m: Shortlist size
        budget: Total number of pulls (including warm start)
        sigma: Standard deviation of reward noise
        true_means: True mean rewards for each arm
        log_every: Log trajectory data every N steps (0 to disable)
    
    Returns:
        Dictionary with:
        - success: Whether best arm is in final shortlist (bool)
        - regret: Simple regret value (float)
        - final_counts: Array of pull counts per arm
        - trajectory: List of (step, success, regret) tuples if log_every > 0
    """
    best_arm = int(np.argmax(true_means))
    
    # Initialize: pull each arm once
    counts, sums, pulls_used = warm_start(rng, k, sigma, true_means)
    remaining_budget = max(budget - pulls_used, 0)
    
    trajectory = []
    total_pulls = pulls_used
    
    for step in range(remaining_budget):
        # Thompson sampling: sample from posterior, pick best
        posterior_means, posterior_variances = posterior_parameters(counts, sums, sigma)
        samples = rng.normal(loc=posterior_means, scale=np.sqrt(posterior_variances))
        chosen_arm = int(np.argmax(samples))  # Pick arm with highest sample

        reward = rng.normal(loc=true_means[chosen_arm], scale=sigma)
        counts[chosen_arm] += 1
        sums[chosen_arm] += reward
        total_pulls += 1
        
        # Log trajectory at intermediate steps
        if log_every > 0 and (step + 1) % log_every == 0:
            shortlist_current = final_shortlist(counts, sums, m)
            success_current = best_arm in shortlist_current
            regret_current = compute_simple_regret(true_means, shortlist_current, counts, sums)
            trajectory.append((total_pulls, success_current, regret_current))
    
    # Final evaluation
    shortlist_final = final_shortlist(counts, sums, m)
    success = best_arm in shortlist_final
    regret = compute_simple_regret(true_means, shortlist_final, counts, sums)
    
    return {
        "success": success,
        "regret": regret,
        "final_counts": counts.copy(),
        "trajectory": trajectory,
    }


def run_a1_sampling(
    rng: np.random.Generator,
    k: int,
    m: int,
    budget: int,
    sigma: float,
    beta_top: float,
    true_means: Sequence[float],
    log_every: int = 10,
) -> Dict:
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
        k: Number of arms
        m: Shortlist size
        budget: Total number of pulls (including warm start)
        sigma: Standard deviation of reward noise
        beta_top: Probability of selecting the best shortlist arm (vs challenger)
        true_means: True mean rewards for each arm (used for generating rewards)
        log_every: Log trajectory data every N steps (0 to disable)
    
    Returns:
        Dictionary with:
        - success: Whether best arm is in final shortlist (bool)
        - regret: Simple regret value (float)
        - final_counts: Array of pull counts per arm
        - trajectory: List of (step, success, regret) tuples if log_every > 0
    """
    best_arm = int(np.argmax(true_means))
    
    # Initialize: pull each arm once
    counts, sums, pulls_used = warm_start(rng, k, sigma, true_means)
    remaining_budget = max(budget - pulls_used, 0)
    
    trajectory = []
    total_pulls = pulls_used
    
    # Main loop: use remaining budget to explore/exploit
    for step in range(remaining_budget):
        # Compute current posterior beliefs about each arm
        posterior_means, posterior_variances = posterior_parameters(counts, sums, sigma)

        # Step 1: Sample from posterior to form shortlist
        # This is Thompson sampling: sample from posterior, pick top m
        samples = rng.normal(loc=posterior_means, scale=np.sqrt(posterior_variances))
        shortlist = np.argsort(samples)[-m:][::-1]  # Top m arms from this sample
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
        if rng.uniform() < beta_top:
            chosen_arm = int(shortlist_best)  # Exploit: use best from shortlist
        else:
            chosen_arm = challenger  # Explore: try the challenger

        # Step 4: Pull chosen arm and update posterior
        reward = rng.normal(loc=true_means[chosen_arm], scale=sigma)
        counts[chosen_arm] += 1
        sums[chosen_arm] += reward
        total_pulls += 1
        
        # Log trajectory at intermediate steps
        if log_every > 0 and (step + 1) % log_every == 0:
            shortlist_current = final_shortlist(counts, sums, m)
            success_current = best_arm in shortlist_current
            regret_current = compute_simple_regret(true_means, shortlist_current, counts, sums)
            trajectory.append((total_pulls, success_current, regret_current))

    # Final evaluation
    shortlist_final = final_shortlist(counts, sums, m)
    success = best_arm in shortlist_final
    regret = compute_simple_regret(true_means, shortlist_final, counts, sums)
    
    return {
        "success": success,
        "regret": regret,
        "final_counts": counts.copy(),
        "trajectory": trajectory,
    }


def run_toptwo_ts(
    rng: np.random.Generator,
    k: int,
    m: int,
    budget: int,
    sigma: float,
    beta: float,
    true_means: Sequence[float],
    log_every: int = 10,
) -> Dict:
    """
    Top-Two Thompson Sampling algorithm.
    
    TopTwoTS samples from the posterior to find the best arm I^(1), then resamples
    until it finds a different arm I^(2) that beats the rest. It then plays I^(1)
    with probability beta, otherwise I^(2).
    
    Algorithm steps per pull:
    1. Sample from posterior to find best arm I^(1)
    2. Resample until finding a different arm I^(2) that beats all others
    3. With probability beta, pull I^(1); otherwise pull I^(2)
    4. Update posterior with observed reward
    
    Args:
        rng: Random number generator
        k: Number of arms
        m: Shortlist size (not used in algorithm, but needed for evaluation)
        budget: Total number of pulls (including warm start)
        sigma: Standard deviation of reward noise
        beta: Probability of selecting I^(1) vs I^(2)
        true_means: True mean rewards for each arm
        log_every: Log trajectory data every N steps (0 to disable)
    
    Returns:
        Dictionary with:
        - success: Whether best arm is in final shortlist (bool)
        - regret: Simple regret value (float)
        - final_counts: Array of pull counts per arm
        - trajectory: List of (step, success, regret) tuples if log_every > 0
    """
    best_arm = int(np.argmax(true_means))
    
    # Initialize: pull each arm once
    counts, sums, pulls_used = warm_start(rng, k, sigma, true_means)
    remaining_budget = max(budget - pulls_used, 0)
    
    trajectory = []
    total_pulls = pulls_used
    
    for step in range(remaining_budget):
        # Compute current posterior beliefs
        posterior_means, posterior_variances = posterior_parameters(counts, sums, sigma)
        
        # Step 1: Sample to find best arm I^(1)
        samples1 = rng.normal(loc=posterior_means, scale=np.sqrt(posterior_variances))
        i1 = int(np.argmax(samples1))
        
        # Step 2: Resample until we find I^(2) != I^(1) that beats all others
        while True:
            samples2 = rng.normal(loc=posterior_means, scale=np.sqrt(posterior_variances))
            i2 = int(np.argmax(samples2))
            if i2 != i1:
                break
        
        # Step 3: Choose between I^(1) and I^(2) based on beta
        if rng.uniform() < beta:
            chosen_arm = i1
        else:
            chosen_arm = i2
        
        # Step 4: Pull chosen arm and update posterior
        reward = rng.normal(loc=true_means[chosen_arm], scale=sigma)
        counts[chosen_arm] += 1
        sums[chosen_arm] += reward
        total_pulls += 1
        
        # Log trajectory at intermediate steps
        if log_every > 0 and (step + 1) % log_every == 0:
            shortlist_current = final_shortlist(counts, sums, m)
            success_current = best_arm in shortlist_current
            regret_current = compute_simple_regret(true_means, shortlist_current, counts, sums)
            trajectory.append((total_pulls, success_current, regret_current))
    
    # Final evaluation
    shortlist_final = final_shortlist(counts, sums, m)
    success = best_arm in shortlist_final
    regret = compute_simple_regret(true_means, shortlist_final, counts, sums)
    
    return {
        "success": success,
        "regret": regret,
        "final_counts": counts.copy(),
        "trajectory": trajectory,
    }

