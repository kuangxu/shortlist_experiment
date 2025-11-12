## Shortlist Experiments

Minimal setup to reproduce the A1 shortlist sampling simulations described in `design.md`.

## Quick Start

```bash
python run_a1_simulation.py --help
python run_a1_simulation.py --replications 100 --debug-every 25
```

The script prints configuration details, runs uniform, Thompson, and A1 sampling baselines, and reports shortlist success rates.

## Implementation Summary

`run_a1_simulation.py` instantiates a Gaussian multi-armed bandit with a warm-start prior, then:
- updates posterior means/variances per pull using the uninformative-prior formulas from `design.md`
- simulates three allocation rules (uniform, Thompson sampling, A1 Algorithm 1)
- repeats each policy for the requested number of replications
- counts how often the final shortlist (top posterior means) contains the true best arm

The objective is to compare shortlist success probabilities under identical budgets and noise assumptions.

## Simulation Setup

- **Reward model:** Each arm emits rewards from `N(theta_i, σ²)` with a shared known variance (`--sigma-sq`).
- **Warm start:** Every policy begins by pulling each arm once (`k` pulls) to initialize posterior counts/sums.
- **Posterior tracking:** Means `μ_i = S_i / N_i`, variances `σ_i² = σ² / N_i`; no priors beyond the warm start.
- **Policies evaluated:**
  - `Uniform Allocation`: random arm each step.
  - `Thompson Sampling`: sample once per arm per step, pull argmax.
  - `A1 Sampling`: Algorithm 1 from `design.md`, β vector with `β₁ = --beta-top`, all others zero.
- **Budget:** `--budget` counts total pulls including the warm start; remaining budget is spent following the policy.
- **Replications:** Repeat the entire run `--replications` times; each replication regenerates the warm start.
- **Success metric:** After budget exhaustion, select shortlist of size `m` via highest posterior means and check if it contains the true best arm.

## Defaults

- Arms `k=10`, shortlist size `m=3`
- Budget includes the `k` warm-start pulls
- Reward variance `σ²=1.0`
- True means default to the example in `design.md`; pass `--true-means` to override

See `design.md` for the theoretical motivation. No additional documentation is maintained beyond the essentials here.

