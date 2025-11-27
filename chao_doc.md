Numerical study for shortlist experiments




Policies
Uniform allocation
Thompson sampling (TS)
A1 sampling with parameters (½, 0, …, 0)
A1 sampling with parameters (β, 0, …, 0), where β = (1 + sqrt{k-1}) / (1 + 3*sqrt{k-1})
TopTwoTS with parameter 1/2
TopTwoTS with parameter β

Note: β is the worst-case parameter, a new result I derived in response to Ramesh’s question. For all allocation rules above, we pair them with the empirical best-arm selection at stopping: we output the shortlist of size m consisting of the arms with the highest empirical means.

Metrics under which even a single sample path can highlight performance differences
Number of measurement required to reach a given confidence level
Posterior probability that each arm is the best
Arm frequency

Metrics that require averaging over many sample paths
(Frequentist) success probability
(Frequentist) simple regret
Arm frequency


Instances
Noise level: sigma^2 = 1
Number of arms: k = 100
Shortlist size: m = 10
Budget: n = 1000

Constructing good instances that highlight performance differences between algorithms requires more thought. Initial ideas for constructing instances include those of the form (1, x, …, x, 0, …, 0), where there are m - 1 arms with unknown mean x and k - m arms with unknown mean 0. (Our theory shows that varying x between (0,1) keeps the asymptotic difficulty unchanged.) Our theory suggests that choosing x close to 1 tends to better reveal the performance gaps between policies, at least when the budget n is large, so I think a good starting choice is x = 0.9.



Future extension: heterogeneous variances


