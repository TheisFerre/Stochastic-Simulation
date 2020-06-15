# Stochastic-Simulation
Git Repo for course in Stochastic Simulation

## Day 1:
Create a Linear Congruential Generator(LCG) to simulate random uniform numbers. Test the numbers with Chi-squared and Kolmogorov-Smirnov test.

The LCG algorithm takes three inputs to (pseudo) randomly sample from a uniform distribution. Below is shown which values was used to create the uniform numbers for this assignment:
```python
numbers = lcg(a=4629, c=17, m=65536, init=2353)
```
With the LCG we generated 10000 numbers. To test the generator we ran multiple test:

```python
plt.hist(numbers, bins=10)
```
![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day1/Histogram_expectation.png)

### Test for correlation
```python
plt.scatter(numbers[:len(numbers)-1], numbers[1:])
```
![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day1/Scatter_plot_corr.png)

### Chi-squared Test
```python
Test_statistic, p_val = chi_squared(numbers, num_classes=10)
```

```
Test_statistic = 5.986
p_val = 0.7413
```

### Kolmogorov-Smirnov Test
```python
Adjusted_test_statistic = kolmogorov(numbers)
```
```
Adjusted_test_statistic = 1.02
```
![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day1/Kolmogorov.png)

### Run-Test III
```python
Test_statistic = run_test(numbers)
```
```
Test_statistic = 0.869
```

## Day 2
This day we simulated both discrete and continous random variables from different distributions using the Uniform distribution.

### Discrete sampling

#### True probabilities:
```python
probabilities = np.array([7/48, 5/48, 1/8, 1/16, 1/4, 5/16])
```

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day2/probs.png)

#### Direct Method:
```python
direct_method(probabilities, plot=True)
```

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day2/direct_sampling.png)

#### Rejection Method:
```python
rejection_method(probabilities, plot=True)
```

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day2/rejection_method.png)

#### Alias Method:
```python
alias_method(probabilities, plot=True)
```

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day2/alias_method.png)

### Continous Sampling

In this section we sampled from continous distributions using the uniform distribution

#### Exponential distribution ```exp(lambda=3)```:

```python
samples_exp = exp_dist(lambd=3, plot=True)
```

Here we also tested our simulated values using the Anderson-Darling test:
```python
test_stat_exp, critical_vals, significance = stats.anderson(samples_exp, dist='expon')
Exponential Test statistic = 0.65182
5_perc_critical_value = 1.341
```

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day2/exponen.png)

#### Normal Distribution ```N(mean=0, variance=1)```:

```python
samples_norm = normal_dist(plot=True)
```

Here we also tested our simulated values using the Kolmogorov-Smirnov test:
```python
test_stat_norm, p_val_norm = stats.kstest(samples_norm, 'norm')
Normal Test statistic: 0.00788
Normal p-value: 0.5632
```
![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day2/normal.png)


We also created 100 confidence intervals from our normal distribution using 10 samples:

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day2/normal_conf.png)

In the plot above we see that it is very noisy and that the confidence intervals are quite wide. This is due to the small sample size of 10, that leads to a lot of noise. We do however see our mean oscillate around mean 0 which is encouraging and expected.

#### Pareto Distribution ```P(beta=1, k=[2.05, 4]```:

We can now compare our sampled pareto values from the true analytical mean/variance:

```python
k = 2.05
beta = 1
emp_mean, emp_var = pareto(beta=beta, k=k, plot=True, moments=True)

analytical_mean = k/(k-1) * beta
analytical_var = k / ((k-1)**2 * (k-2)) * beta**2
```

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day2/pareto_k205.png)

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day2/pareto_k4.png)

## Day 3

This day we was supposed to simulate "Block system". This system contains ```n``` service units, mean service unit time, mean customer arrival.
The system simulates customers that show up to a store to get a service from one of the ```n``` service units. The Service units mean availability is modelled with an exponential function. Customer arrival mean is modeled as a Poisson process.

```python
num_service_units = 10
mean_service_time = 8
mean_customer_arrival = 1

system = BlockSystem(
    num_service_units=num_service_units,
    mean_service_time=mean_service_time,
    mean_customer_arrival=mean_customer_arrival,
    customer_arrival_dist='expon',  # expon/erlang/hyperexp
    service_time_dist='expon'  # expon/pareto
)
```
System Dynamics:

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day3/blocking_system.png)

95% Confidence interval on fraction blocked/serviced

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day3/confidence_blocking.png)

## Day 4

This day we discovered various variance reduction methods. As an example we used the control variate method to reduce variance on the final blocking fraction of our blocking system from day 3.

We found that the original variance was ```1.07e-05``` but after variance reduction it was reduced to ```4.831```

Additionally, we used the  Common Random method for comparing two different simulations of our blocking system. Here we used the two following simulations:

```python
## SIMULATION 1
# Hyper Exponential Arrival times

num_service_units_hyper = 10
mean_service_time_hyper = 1
mean_customer_arrival_hyper = [(0.8, 0.8333), (0.2, 5)]

system_hyper = BlockSystem(
    num_service_units=num_service_units_hyper,
    mean_service_time=mean_service_time_hyper,
    mean_customer_arrival=mean_customer_arrival_hyper,
    customer_arrival_dist='hyperexp',
    service_time_dist='expon'
)

## SIMULATION 2
# Exponential Arrival times
num_service_units_exp = 10
mean_service_time_exp = 8
mean_customer_arrival_exp = 1

system_exp = BlockSystem(
    num_service_units=num_service_units_exp,
    mean_service_time=mean_service_time_exp,
    mean_customer_arrival=mean_customer_arrival_exp,
    customer_arrival_dist='expon',  # expon/erlang/hyperexp
    service_time_dist='expon'  # expon/pareto
)
```



The estimator for the difference of the two systems is given by <img src="https://bit.ly/30IFcE4" align="center" border="0" alt="\hat{\theta}_{2} - \hat{\theta}_{1}" width="61" height="22" />, where each <img src="https://bit.ly/3e7Jhpm" align="center" border="0" alt="\hat{\theta}" width="17" height="19" /> is computed by simulating it several times. An important feature of the common random numbers method, is that the same (pseudo) randomly generated numbers are used for both simulation such that they can be compared properly eliminating difference that could come from randomness.

Using a 95% confidence interval we found: <img src="https://bit.ly/3e2FxFN" align="center" border="0" alt="\hat{\theta}_{2} - \hat{\theta}_{1}=0.117 \pm  0.0023" width="214" height="22" />


**Day5

This day we discovered the Markov Chain Monte Carlo (MCMC) method for estimating distributions.

First we used the following function for MCMC:
<img src="https://bit.ly/37zbMdg" align="center" border="0" alt="P(i) = \frac{\frac{A^{i}}{i!}} {\sum_{j=0}^{N} \frac{A^{j}}{j!} }, \quad j=0 \dots N" width="219" height="65" />, letting N=10 and alpha=3.

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day5/MCMC1.png)












