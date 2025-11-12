Here is a simple guide on how to implement the A1 sampling algorithm and set up a basic experiment to test it, based on the provided document.

## 1. How to Implement A1 Sampling

This guide assumes you are working with a known number of arms, `k`, and want to find a shortlist of size `m`. [cite_start]The algorithm is designed for an environment where each arm `i` provides Gaussian rewards with an unknown mean $\theta_i$ and a known, common variance $\sigma^2$[cite: 271].

### 📋 Setup & Initialization

1.  **Define Parameters:**
    * `k`: Total number of arms (e.g., 10).
    * `m`: Size of the desired shortlist (e.g., 3).
    * [cite_start]`\sigma^2`: The known variance of rewards for all arms (e.g., 1.0)[cite: 271].

2.  **Set Algorithm Tuning (`\bm{\beta}`):**
    * [cite_start]The algorithm uses a probability vector $\bm{\beta}$ to decide which arm to sample from the "And1" set[cite: 232, 240].
    * [cite_start]A simple and effective choice recommended in the paper is to set `\beta_1 = 0.5` and all other $\beta_i = 0$ (for $i=2, \ldots, m$)[cite: 250].
    * This choice simplifies the sampling step (Step 4, below) significantly: you will only ever sample between the *best* arm of a potential shortlist and one "challenger" arm.

3.  **Initialize Posterior Data:**
    * For each arm `i \in \{1, \ldots, k\}`, create two trackers:
        * `N_i`: The number of times arm `i` has been pulled.
        * `S_i`: The sum of rewards received from arm `i`.
    * [cite_start]**Warm Start:** To avoid division by zero (as the math assumes an uninformative prior [cite: 286]), pull each of the `k` arms one time. For each arm `i`:
        * Observe reward `Y_i`.
        * Set `N_i = 1` and `S_i = Y_i`.

### 🔄 The A1 Sampling Loop

Run this loop for your desired number of steps (your total budget, `T`).

**At each step `t`:**

1.  **Get Current Posteriors:**
    * For each arm `i`, calculate its current posterior distribution, which is $N(\mu_i, \sigma_i^2)$:
        * [cite_start]Posterior Mean: $\mu_i = S_i / N_i$[cite: 288].
        * [cite_start]Posterior Variance: $\sigma_i^2 = \sigma^2 / N_i$[cite: 288].

2.  **Step 1: Find the Leading Shortlist (`S_t`)**
    * [cite_start]Draw one sample, $\tilde{\theta}_i$, from each arm's posterior distribution $N(\mu_i, \sigma_i^2)$[cite: 233].
    * [cite_start]Find the indices of the `m` arms that have the highest sample values[cite: 234].
    * [cite_start]This set of `m` indices is your **leading shortlist, `S_t`**[cite: 234].
    * [cite_start]Identify the index of the *best* arm in this list: `J_{t,1}` (the arm with the highest $\tilde{\theta}_i$ among the `m` arms)[cite: 234].

3.  **Step 2: Find the Challenger (`L_t`)**
    * [cite_start]Start a resampling loop (e.g., a `while True:` loop)[cite: 236].
    * [cite_start]Inside the loop: Draw a *new* set of samples, $\tilde{\theta}'_i$, from all `k` posterior distributions $N(\mu_i, \sigma_i^2)$[cite: 236].
    * [cite_start]Find the index `L_t` of the arm with the single highest sample: $L_t = \argmax_i \tilde{\theta}'_i$[cite: 237].
    * [cite_start]**Check:** If this challenger arm `L_t` is *not* in your leading shortlist `S_t`, break the loop[cite: 237]. Otherwise, repeat the resampling.

4.  **Step 3: Choose Arm to Pull**
    * You now have two arms to choose from: `J_{t,1}` (the best from the shortlist) and `L_t` (the challenger).
    * Using the simple $\bm{\beta}$ from the setup ($\beta_1 = 0.5$):
        * With 50% probability, choose $I_t = J_{t,1}$.
        * With 50% probability, choose $I_t = L_t$.
    * [cite_start](The formal algorithm calls this combined set the "And1 set" $S_t^{+} = S_t\cup \{L_t\}$ [cite: 239]).

5.  **Step 4: Observe and Update**
    * [cite_start]Pull the chosen arm `I_t` and observe its reward `Y_t`[cite: 240].
    * Update your posterior data for that arm:
        * $N_{I_t} = N_{I_t} + 1$
        * $S_{I_t} = S_{I_t} + Y_t$
    * [cite_start]This update step implicitly updates the posterior $\Pi_{t}\to \Pi_{t+1}$[cite: 240].

### 🏁 Final Output (After the Loop)

After your total budget `T` is exhausted, you must output a final shortlist.
* [cite_start]**Bayes-Optimal Rule:** The formal method is to select the shortlist `S` that maximizes the posterior probability of containing the best arm[cite: 219, 225].
* [cite_start]**Simple/Natural Rule:** A more natural decision rule, which the paper notes the Bayes-optimal rule converges to, is to simply select the `m` arms with the **highest posterior means** ($\mu_i = S_i / N_i$)[cite: 229].

---

## 2. How to Run a Simple Experiment

Here’s how to set up a simulation to evaluate the performance of your A1 implementation.

### 🌎 Environment Setup

1.  [cite_start]**Define Ground Truth:** You (the experimenter) must define the *true* unknown means for all arms: $\thetabf = (\theta_1, \theta_2, \ldots, \theta_k)$[cite: 180]. This is what the algorithm is trying to learn.
    * **Example:** For `k=10` and `m=3`, you could set:
        `\thetabf = (0.5, 0.45, 0.4, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1)`
    * [cite_start]In this example, the true best arm `I^*` is arm 1 (with mean 0.5)[cite: 281].
    * [cite_start]The true "top-m" set is `S_top = \{1, 2, 3\}`[cite: 282].

2.  **Define Experiment Parameters:**
    * `T`: Total budget (e.g., `T = 5000` pulls).
    * `R`: Number of replications (e.g., `R = 1000`). You need to run the whole experiment many times to get a stable average.
    * [cite_start]`\sigma^2`: Reward variance (e.g., 1.0)[cite: 271].

3.  [cite_start]**Define Reward Simulator:** Create a function `get_reward(i)` that, when called, returns a sample from $N(\theta_i, \sigma^2)$, using your ground truth $\theta_i$[cite: 271].

### 📊 Running the Simulation

1.  **Baselines:** To see if A1 sampling is effective, compare it against:
    * **Uniform Allocation (RCT):** At each step `t`, pick an arm `I_t` uniformly at random from $\{1, \ldots, k\}$.
    * **Standard Thompson Sampling (TS):** At each step `t`, draw one sample $\tilde{\theta}_i$ from each posterior and pull the arm with the highest sample, $I_t = \argmax_i \tilde{\theta}_i$.

2.  **Experiment Loop:**
    * Initialize a "success counter" to 0.
    * **For `r = 1` to `R` (replications):**
        1.  Reset your algorithm (clear all `N_i` and `S_i` counters).
        2.  Run the **A1 Sampling** algorithm (including the warm start) for `T` steps, using your `get_reward()` function.
        3.  [cite_start]**Get Final Shortlist:** After `T` steps, identify the final shortlist `S_T` using the "Simple/Natural Rule" (the `m` arms with the highest posterior means $\mu_i$)[cite: 229].
        4.  **Check for Success:** Check if the *true best arm* (`I^*`, which is arm 1 in our example) is in your final shortlist `S_T$. [cite_start]This check corresponds to the event $\max_{i\in S_T}\tilde{\theta}_i = \max_{j\in [k]}\tilde{\theta}_j$ that the objective function tries to maximize[cite: 203, 225].
        5.  If `I^* \in S_T`, increment your success counter.

3.  **Calculate Performance:**
    * **Success Rate = (Success Counter) / R**
    * This metric tells you the probability that your algorithm successfully includes the best possible arm in its final recommendation.

4.  **Compare:** Repeat steps 2-3 for your baseline algorithms (Uniform and TS). You can then plot the Success Rate vs. Budget (`T`) for all three algorithms to see which one learns to identify the best arm most efficiently.

---

## 3. Formal Math Reference (Appendix)

Here are the formal definitions from the paper corresponding to the implementation guide.

### 📐 Problem Formulation

* [cite_start]**Set of Shortlists ($\Sc$):** The collection of all feasible shortlists (subsets of size `m`)[cite: 189].
    $$ \Sc \triangleq \left\{S\subseteq [k]: |S| = m\right\} $$
* [cite_start]**Loss Function ($\ell$):** A function that penalizes a shortlist `S` if the best arm (under a posterior sample $\tilde{\bm\theta}$) is not in `S`[cite: 202].
    $$ \ell(S,\tilde{\bm\theta}) = \ind\left\{\max_{i\in S}\tilde{\theta}_i < \max_{j\in [k]}\tilde{\theta}_j\right\} $$
* **Adaptive Posterior Risk ($\mathfrak{E}$):** The objective function to be minimized. [cite_start]It is the posterior expected loss, conditional on the collected data (history `H`)[cite: 203, 204].
    $$ \mathfrak{E}_{(H, S)} = \E_{H}\left[\ell(S,\tilde{\bm\theta})\right] = 1 - \Prob_{H}\left(\max_{i\in S}\tilde{\theta}_i = \max_{j\in [k]}\tilde{\theta}_j \right) $$

### 🎯 Bayes-Optimal Shortlisting

* [cite_start]**Posterior Probabilities ($\alpha$):** Given history $H_t$, $\alpha_{H_t,i}$ is the posterior probability that arm `i` is the best[cite: 223]. [cite_start]$\alpha_{H_t,S}$ is the probability that the best arm is in the set `S`[cite: 225].
    $$ \alpha_{H_t,i}\triangleq\Prob_{H_t}\left(\tilde{\theta}_i = \max_{j\in[k]}\tilde{\theta}_j \right) $$
    $$ \alpha_{H_t,S} \triangleq \sum_{i\in S} \alpha_{H_t,i} = \Prob_{H_t}\left(\max_{i\in S}\tilde{\theta}_i = \max_{j\in[k]}\tilde{\theta}_j \right) $$
* [cite_start]**Bayes-Optimal Shortlist ($S^*_{H_t}$):** The shortlist that minimizes the posterior risk [cite: 219][cite_start], or equivalently, maximizes $\alpha_{H_t,S}$[cite: 225].
    $$ S^*_{H_t} \in \argmin_{S\in\mathcal{S}}\mathfrak{E}_{(H_t,S)} $$

### 🤖 A1 Sampling (Algorithm 1)

[cite_start]**Input:** prior $\Pi_0$, parameters $\bm{\beta} \in\mathcal{B}$ [cite: 232]

**For $t = 0,1,\ldots$**
1.  [cite_start]Sample $\tilde{\bm{\theta}} = \left(\tilde{\theta}_1,\ldots, \tilde{\theta}_k\right)$ from the posterior $\Pi_{t}$[cite: 233].
2.  [cite_start]Form the leading shortlist $S_t\triangleq \{{J_{t,1}, \ldots, J_{t,m}}\}$, where ${J}_{t,1}, \ldots, {J}_{t,m}$ denote the indices of the best $m$ arms under $\tilde{\bm{\theta}}$[cite: 234].
3.  [cite_start]Resample until obtaining $\tilde{\bm{\theta}}' = \left(\tilde{\theta}_1',\ldots, \tilde{\theta}_k'\right)$ such that the best arm under $\tilde{\bm{\theta}}'$ falls outside the leading shortlist[cite: 236]:
    [cite_start]$$ \exists L_t\in\argmax_{i\in [k]} \tilde{\theta}_i' \quad\text{such that} \quad L_t \notin S_t $$ [cite: 237]
4.  [cite_start]Form the And1 set $S_t^{+} = S_t\cup \{L_t\}$[cite: 239].
5.  [cite_start]Choose an arm from the And1 set[cite: 240]:
    $$ I_t \in S_t^{+} \quad\text{with probabilities}\quad \left(\beta_{1},\ldots, \beta_{m}, 1 -\sum_{i=1}^m \beta_{i}\right) $$
6.  [cite_start]Observe the reward $Y_{t+1,I_t}$ from the chosen arm $I_t$ and update the posterior $\Pi_{t}\to \Pi_{t+1}$[cite: 240].
**End For**

### ⚙️ Assumptions for Theoretical Results

* [cite_start]**Gaussian Observations:** For each arm $i$, rewards are drawn from $P(\cdot \,|\, \theta_i) = \mathcal{N}(\theta_i, \, \sigma^2)$ where the variance $\sigma^2$ is common[cite: 271].
* [cite_start]**Uninformative Priors:** We assume independent Gaussian priors with infinite variance[cite: 286]. This simplifies the posterior calculation:
    * [cite_start]$N_{t,i}$ = number of samples for arm $i$ up to time $t$[cite: 287].
    * [cite_start]Posterior Mean: $\theta_{t,i} \triangleq \frac{\sum_{\ell=0}^{t-1}{\bm{1}(I_{\ell} = i)Y_{\ell+1, I_{\ell}}}}{N_{t,i}}$[cite: 288].
    * [cite_start]Posterior Variance: $\sigma^2_{t,i} \triangleq \frac{\sigma^{2}}{N_{t,i}}$[cite: 288].

