Here is the simulation plan based on the provided documents.

# Simulation Plan: Shortlist Selection Experiments

## 1. Experimental Setup

### 1.1 Global Parameters
These parameters remain fixed across all simulations to ensure comparability.
* **Total Arms ($k$):** $100$
* **Shortlist Size ($m$):** $10$
* **Budget ($n$):** $1000$ pulls
* **Noise Variance ($\sigma^2$):** $1$ (Standard Normal noise)

### 1.2 Instance Construction (The "Hard" Instance)
To highlight performance differences between policies, we will use a specific "hard" instance structure rather than random means.
* **Mean Vector Structure:** $(1, x, \dots, x, 0, \dots, 0)$
* **Difficulty Parameter ($x$):** $0.9$
    * *Reasoning:* Theory suggests choosing $x$ close to 1 tends to better reveal performance gaps between policies when the budget is large.
* **Arm Breakdown:**
    * **1 Best Arm:** $\mu = 1.0$
    * **$m-1$ Distractor Arms:** $\mu = 0.9$ (9 arms)
    * **$k-m$ Noise Arms:** $\mu = 0.0$ (90 arms)

### 1.3 Stopping & Selection Rule
* **Stopping:** The experiment concludes exactly at $n=1000$.
* **Final Output:** The shortlist of size $m$ consisting of the arms with the highest **empirical means** at step $n$.

---

## 2. Policies to Evaluate

We will simulate six distinct policy configurations.

| Policy | $\beta$ Parameter | Description |
| :--- | :--- | :--- |
| **Uniform** | N/A | Uniform Random Allocation. |
| **Thompson Sampling (TS)** | N/A | Standard greedy TS (select arm with highest posterior sample). |
| **A1 (Standard)** | $0.5$ | A1 sampling with parameter $(1/2, 0, \dots, 0)$. |
| **A1 (Optimal)** | $\beta^*$ | A1 sampling with derived worst-case parameter. |
| **TopTwoTS (Standard)** | $0.5$ | Top-Two Thompson Sampling with parameter $1/2$. |
| **TopTwoTS (Optimal)** | $\beta^*$ | Top-Two TS with derived worst-case parameter. |

### Parameter Calculation
The worst-case parameter $\beta^*$ (labeled "Optimal" above) is derived as:
$$\beta^* = \frac{1 + \sqrt{k-1}}{1 + 3\sqrt{k-1}}$$
For $k=100$:
$$\beta^* = \frac{1 + \sqrt{99}}{1 + 3\sqrt{99}} \approx \frac{10.95}{30.85} \approx 0.355$$
This parameter applies to both the A1 "Optimal" and TopTwoTS "Optimal" variants.

---

## 3. Metrics

### 3.1 Aggregate Metrics (Averaged over $R$ paths)
We will run $R=1000$ replications to smooth noise.
1.  **Frequentist Success Probability:** The fraction of runs where the true best arm (index 0, $\mu=1.0$) is contained in the final shortlist.
2.  **Simple Regret:** The difference between the best arm's mean and the highest mean in the selected shortlist.
    * If Best Arm $\in$ Shortlist: Regret $= 0$.
    * If Best Arm $\notin$ Shortlist (but distractors are): Regret $= 1.0 - 0.9 = 0.1$.
3.  **Arm Frequency:** The average number of samples allocated to:
    * The Best Arm.
    * The Distractor Arms (avg).
    * The Noise Arms (avg).

### 3.2 Single Path Metrics
To visualize trajectory behavior, we will record the following at every time step $t$ for a single representative run:
1.  **Posterior Probability:** The probability that each arm is the best arm.
2.  **Number of Measurements:** The cumulative count $N_{t,i}$ for the best arm.

---

## 4. Implementation Plan

The current codebase (`run_a1_simulation.py`) requires specific updates to support this experiment.

### 4.1 Required Code Changes
1.  **Implement TopTwoTS:** Add a new runner function for Top-Two Thompson Sampling.
    * *Logic:* Sample posterior to find best arm $I^{(1)}$. Resample until a different arm $I^{(2)}$ beats the rest. Play $I^{(1)}$ with prob $\beta$, else $I^{(2)}$.
2.  **Implement Instance Generator:** Add a function to generate the $(1, 0.9, \dots, 0)$ means vector.
3.  **Time-Series Logging:** Modify the loop to store success/regret at *intermediate* steps (e.g., every 10 steps) to plot learning curves, rather than just the final result at $T$.

### 4.2 Output Files
The simulation script should generate:
* `results_summary.csv`: Final success rates and regret for all 6 policies.
* `trajectory_data.json`: Time-series data for learning curves (Success Rate vs Time).
* `allocation_data.csv`: Final pull counts per arm group.

---

## 5. Figures to Produce

### Figure 1: Learning Curve (Success Probability)
* **X-Axis:** Budget consumed ($0$ to $1000$).
* **Y-Axis:** Probability of including Best Arm in Shortlist.
* **Content:** 6 curves (one for each policy).
* **Hypothesis:** A1 (Optimal) should approach 1.0 faster than TS and Uniform.

### Figure 2: Allocation Efficiency
* **Type:** Grouped Bar Chart.
* **Groups:** Best Arm, Distractor Arms, Noise Arms.
* **Y-Axis:** Average Sample Count.
* **Content:** Bars for TS, A1 (Standard), A1 (Optimal).
* **Hypothesis:** A1 should suppress sampling of "Noise" arms faster than Uniform, while distinguishing "Best" from "Distractor" more efficiently than standard TS.

### Figure 3: Posterior Evolution (Single Path)
* **X-Axis:** Time $t$.
* **Y-Axis:** Posterior probability $P(\text{Arm } i = \text{Best})$.
* **Content:** Traces for the Best Arm and one Distractor Arm.
* **Hypothesis:** Shows the "confidence" accumulation speed differences.