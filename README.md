# Stochastic-Simulation Exercises


# Day 1:
**(EXERCISE 1)**

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

## Test for correlation
```python
plt.scatter(numbers[:len(numbers)-1], numbers[1:])
```
![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day1/Scatter_plot_corr.png)

## Chi-squared Test
```python
Test_statistic, p_val = chi_squared(numbers, num_classes=10)
```

```
Test_statistic = 5.986
p_val = 0.7413
```
(Can not reject null hypothesis using 5% significance level)

## Kolmogorov-Smirnov Test
```python
test_statistic, p_val = kolmogorov(numbers)
```
```
adjusted_test_statistic = 1.0206
```
As the Adjusted test statistic for the kolmogorov test is below the 95% critical value of 1.358 we can not reject the null hypothesis.

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day1/kolmogorov_suprema.png)

The green line indicates where the suprema between the analytical and empirical distribution is found.

## Run-Test II
Testing the Run-TestII implementation on the list ```[0.54, 0.67, 0.13, 0.89, 0.33, 0.45, 0.90, 0.01, 0.45, 0.76, 0.82, 0.24, 0.17]```, we find the run-vector [2, 2, 1, 1, 0, 0].
```python
Test_statistic = run_test2(numbers)
```
```
Test_statistic = 5.349
```
The Test statistic is above the 95% critical value of 1.6353 which is found from the Chi-squared distribution. Therefore we reject the null hypothesis.

## Run-Test III
```python
Test_statistic = run_test(numbers)
```
```
Test_statistic = 0.869
```
As the 95% critical value for the normal distribution with mean 0 and variance 1 is 1.64, we can not reject the null hypothesis as the test statistic is below the critical value.

## Correlation Test
```python
# Using h=5
correlation = corr_correlation_test(number, 5)
correlation = 6.735828271520318e-07
```

We also used these test on random numbers generated by python's Random module. Please have a look at the code for this.

# Day 2
**(EXERCISE 2)**

## Geometric distribution

To test the Geometric Distribution we used a Chisquared test, and limited the test to values where the expected frequencies were above 5.

We found that the p-value was varying a lot even though we set the constraint of only picking values where the expected frequencies were above 5. An example p-value was ```0.2689```, where we would not reject our null-hypothesis using a 5% significance level.

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day2/geometric.png)

## Discrete sampling

### True probabilities:
```python
probabilities = np.array([7/48, 5/48, 1/8, 1/16, 1/4, 5/16])
```

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day2/probs.png)

### Direct Method:
```python
direct_method(probabilities, plot=True)
```

Direct Test statistic: ```6.3847```

Direct pval: ```0.27056```

(Can not reject null hypothesis using 5% significance level)


![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day2/direct_sampling.png)

### Rejection Method:
```python
rejection_method(probabilities, plot=True)
```

Reject Test statistic: ```2.2819```

Reject pval: ```0.80893```

(Can not reject null hypothesis using 5% significance level)

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day2/rejection_method.png)

### Alias Method:
```python
alias_method(probabilities, plot=True)
```

Alias Test statistic: ```11.0541```

Alias pval: ```0.05032```

(Can not reject null hypothesis using 5% significance level)

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day2/alias_method.png)

## Continous Sampling
**(EXERCISE 3)**

In this section we sampled from continous distributions using the uniform distribution

### Exponential distribution lambda=3:

```python
samples_exp = exp_dist(lambd=3, plot=True)
```

Here we also tested our simulated values using the Anderson-Darling test:
```python
test_stat_exp, critical_vals, significance = stats.anderson(samples_exp, dist='expon')
Exponential Test statistic = 0.65182
5_perc_critical_value = 1.341
```

Test statistic is below critical value so we can not reject the 5% significance level.

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day2/exponen.png)

### Normal Distribution 
```N(mean=0, variance=1)```

```python
samples_norm = normal_dist(plot=True)
```

Here we also tested our simulated values using the Kolmogorov-Smirnov test:
```python
test_stat_norm, p_val_norm = stats.kstest(samples_norm, 'norm')
Normal Test statistic: 0.00788
Normal p-value: 0.5632
```

(Can not reject null hypothesis using 5% significance level)

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day2/normal.png)


We also created 100 confidence intervals from our normal distribution using 10 samples:

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day2/normal_conf.png)

In the plot above we see that it is very noisy and that the confidence intervals are quite wide. This is due to the small sample size of 10, that leads to a lot of noise. We do however see our mean oscillate around mean 0 which is encouraging and expected.

### Pareto Distribution 
```P(beta=1, k=[2.05, 2.5, 3, 4]```:

We can now compare our sampled pareto values from the true analytical mean/variance:

```python
k = 2.05
beta = 1
emp_mean, emp_var = pareto(beta=beta, k=k, plot=True, moments=True)

analytical_mean = k/(k-1) * beta
analytical_var = k / ((k-1)**2 * (k-2)) * beta**2
```

For k=2.05

Absolute difference emp. vs analytical mean: ```0.02186```

Absolute difference emp. vs analytical variance: ```33.3599```

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day2/pareto_k205.png)

For k=2.5

Absolute difference emp. vs analytical mean: ```0.00686```

Absolute difference emp. vs analytical variance: ```0.5387```

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day2/pareto_k2.png)

For k=3

Absolute difference emp. vs analytical mean: ```0.00505```

Absolute difference emp. vs analytical variance: ```0.0662```

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day2/pareto_k3.png)

For k=4

Absolute difference emp. vs analytical mean: ```0.00845```

Absolute difference emp. vs analytical variance: ```0.04415```

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day2/pareto_k4.png)

We note that the variance when ```k=2.05``` is greater than for other k-values. This comes from the fact that a k value close to 2 will be very sensitive to small changes in the sampled value. This result can also be interpreted from the equations in slide 13 in slideshow ***Sampling from Continuous Distributions***

# Day 3
**(EXERCISE 4)**

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

```python
# erlang/pareto
num_service_units = 10
mean_service_time = 1.1
mean_customer_arrival = 1

system = BlockSystem(
    num_service_units=num_service_units,
    mean_service_time=mean_service_time,
    mean_customer_arrival=mean_customer_arrival,
    customer_arrival_dist='erlang',  # expon/erlang/hyperexp
    service_time_dist='pareto'  # expon/pareto
)

blocked, serviced, service_avail = system.simulate(
    simulation_customers=10000, plot=True)

system(simulations=10, simulation_customers=10000, plot=True)
```

System Dynamics:

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day3/erlang_pareto_blocking.png)

95% Confidence interval on fraction blocked/serviced

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day3/erlang_pareto_conf.png)

```python
# Hyper Exponential/pareto
num_service_units = 10
mean_service_time = 1
mean_customer_arrival = [(0.8, 0.8333), (0.2, 5)]
system = BlockSystem(
    num_service_units=num_service_units,
    mean_service_time=mean_service_time,
    mean_customer_arrival=mean_customer_arrival,
    customer_arrival_dist='hyperexp',
    service_time_dist='pareto'
)


blocked, serviced, service_avail = system.simulate(
    simulation_customers=10000, plot=True)

system(simulations=10, simulation_customers=10000, plot=True)
```

System Dynamics:

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day3/hyper_pareto_blocking.png)

95% Confidence interval on fraction blocked/serviced

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day3/hyper_pareto_conf.png)

# Day 4
**(EXERCISE 5)**

In the following section we estimate the integral provided in the exercise.

True Evaluation of integral: 1.718281828459045
## Monte Carlo estimation
(Refer to slide 3 in slideshow ***Variance Reduction Methods***)
```
Estimation: 1.78334
Confidence Interval: 1.7834+--0.08096
finish time: 0.0014660358428955078
```


## Anithetic Variables estimation
(Refer to slide 5 in slideshow ***Variance Reduction Methods***)
```
Estimation: 1.7149
Confidence Interval: 1.7149+--0.0103
finish time: 0.0004683
```


## Control Variate estimation
(Refer to slide 10 in slideshow ***Variance Reduction Methods***)
```
Estimation: 1.7220
Confidence Interval: 1.7220+--0.0116
finish time: 0.0003616809844970703
```


## Stratified Sampling estimation
(Refer to slide 12 in slideshow ***Variance Reduction Methods***)
```
Estimation: 1.7155
Confidence Interval: 1.7155+--0.00259
finish time: 0.004197359085083008
```

In general all the methods closely estimate the true value of the integral. It should however be noted that there is significant difference in finishing times.

This day we discovered various variance reduction methods. As an example we used the control variate method to reduce variance on the final blocking fraction of our blocking system from day 3.

We found that the original variance was ```4.4571e-05``` but after variance reduction it was reduced to ```2.02835e-05```

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



The estimator for the difference of the two systems is given by <img src="https://render.githubusercontent.com/render/math?math=\hat{\theta}_{2} - \hat{\theta}_{1}">, where each <img src="https://render.githubusercontent.com/render/math?math=\hat{\theta}"> is computed by simulating it several times. An important feature of the common random numbers method, is that the same (pseudo) randomly generated numbers are used for both simulation such that they can be compared properly eliminating difference that could come from randomness.

Using a 95% confidence interval we found: <img src="https://render.githubusercontent.com/render/math?math=\hat{\theta}_{2} - \hat{\theta}_{1}=0.117 \pm  0.0023">


# Day5

## MCMC
**(EXERCISE 6)**

This day we discovered the Markov Chain Monte Carlo (MCMC) method for estimating distributions.

First we used the following function for MCMC:
<img src="https://render.githubusercontent.com/render/math?math=P(i) = \frac{\frac{A^{i}}{i!}} {\sum_{j=0}^{N} \frac{A^{j}}{j!} }, \quad j=0 \dots N">, letting N=10 and A=8.

The proposal distribution was a random walk in the discrete space, where there was a 50% chance of increasing or decreasing our newly proposed position by 1.

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day5/MCMC_1_var.png)

Chi-squared Test: ```p-value=0.0116```

(Reject null hypothesis using 5% significance level)


Next we extended the function to:
<img src="https://render.githubusercontent.com/render/math?math=P(i,j) = \frac{1}{K} \frac{A_{1}^{i}}{i!}\frac{A_{2}^{j}}{j!}, \quad 0\leq i+j \leq N">, where the term 1/K is removed as this is a normalization constant that we will not approximate. 

The proposal distribution was a random walk in the discrete space, where there was a 50% chance of increasing or decreasing our newly proposed position by 1. This was done in both directions while staying within the constraints.

Using A1=4 and A2=4 we get the following:

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day5/MCMC_2_var.png)

Chi-squared Test (X-values): ```pvalue=2.115e-19```

(Reject null hypothesis using 5% significance level)

Chi-squared Test (Y-values): ```pvalue=3.231e-15```

(Reject null hypothesis using 5% significance level)

## GIBBS SAMPLING

We now found conditional distributions for the function given in previous simulation. The two conditional distributions were analytically found to be:

<img src="https://render.githubusercontent.com/render/math?math=P(i | j) = \frac{\frac{A_{1}^{i}}{i!} \frac{A_{2}^{j}}{j!}}{\sum_{i=0}^{N-j} \frac{A_{1}^{i}}{i!} \frac{A_{2}^{j}}{j!} }">

<img src="https://render.githubusercontent.com/render/math?math=P(j | i) = \frac{\frac{A_{1}^{i}}{i!} \frac{A_{2}^{j}}{j!}}{\sum_{j=0}^{N-i} \frac{A_{1}^{i}}{i!} \frac{A_{2}^{j}}{j!} }">

Similarly to the MCMC runs above, we use a proposal distribution where we randomly increase/decrease a single variable by 1. The probability for accepting the proposed position is given by the conditionals. For the Gibbs sampling procedure, we follow this step where we alternate which variable we change.


```python
alpha1 = 4
alpha2 = 4
n = 10
model = MCMC_gibs(func1=p_i_given_j, func2=p_j_given_i,
                  num_classes=n, position=(int(n/2)-1, int(n/2)-1))
samples = model.run(num_samples=1000)
```

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day5/GIBBS_hist.png)

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day5/gibbs_grid.png)

Chi-squared Test (X-values): ```pvalue=1.2471-86```

(Reject null hypothesis using 5% significance level)

Chi-squared Test (Y-values): ```pvalue=2.732e-53```

(Reject null hypothesis using 5% significance level)


## TSP Simulated Annealing
**(EXERCISE 7)**

We now used Simulated annealing to minimize cost on a TSP problem given a cost matrix over 20 cities. Simulated Annealing is inspired by statistical physics and is supposed to find a global optimum by using "enery" of states. The temperature function we used was <img src="https://render.githubusercontent.com/render/math?math=T(k) = \frac{1}{\sqrt{1+k}}">. The value of k was changed every 10th iteration of our optimization with the following scheme: <img src="https://render.githubusercontent.com/render/math?math=K =\frac{1}{10}">.

As a proof of concept we created a circle where we used our optimization scheme to find the minimum cost (distance) for the circle. The green line represent the initial (random) TSP, while the blue line is our optimized TSP:


![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day5/circle_tsp.png)

The optimization of the circle worked fine. We now proceed to using the cost matrix to find the global minimum for the cost of the given problem.

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day5/TSP_end_cost.png)

For the tsp with a given cost-matrix we repeated the simulation 10 times to compute a 95% confidence interval for the end-cost. For the 10 simulations we found the 95% confidence intervals to be ```0.8875 +- 0.0345```.


# Day 6
**(EXERCISE 8)**

This day we used bootstrapping to estimate mean/median/variance of randomly distributed variables. 

## Exercise 1
Bootstrap approach to estimate p by constructing random resamples of X_i, ..., X_n.
```python
n = 10
X = [56, 101, 78, 67, 93, 87, 64, 72, 80, 69]
a = -5
b = 5

p_list = []


def func1(n=10, X=X, a=a, b=b):

    bootstrap_vals = np.random.choice(X, n)
    x_sum = 0
    for i in range(n):
        x_sum += bootstrap_vals[i]/n

    x_sum = x_sum - np.mean(X)

    if a < x_sum and x_sum < b:
        return 1
    else:
        return 0


p_vals = []
for i in range(100):
    p_vals.append(func1())

print('Exercise 1')
print(np.mean(p_vals))
```

The mean was found to be ```0.75```

## Exercise 2

```python
def var(X, mean):

    sum_var = 0

    for i in range(len(X)):
        sum_var += (X[i] - mean)**2

    return sum_var/(len(X)-1)


X = [5, 4, 9, 6, 21, 17, 11, 20, 7, 10, 21, 15, 13, 16, 8]

var_list = []
for i in range(100):

    # Pick from X 15 times.
    pick = np.random.choice(X, size=15)

    var_list.append(var(pick, np.mean(X)))

print('Exercise 2')
print(np.var(var_list))
```

The variance was found to be ```70.422```

## Exercise 3

```python
def get_samples(num_samples):
    # Sample several times from the pareto dist.
    samples = stats.pareto.rvs(1.05, size=num_samples)

    return samples


mean_list = []
median_list = []
for i in range(100):
    samples = get_samples(200)

    mean_list.append(np.mean(samples))
    median_list.append(np.median(samples))

print('Exercise 3')
print('mean (mean/var')
print(np.mean(mean_list))
print(np.var(mean_list))

print('median (mean/var')
print(np.mean(median_list))
print(np.var(median_list))
```

The mean and variance of the mean was found to be ```8.194``` and ```45.477```, respectively.

The mean and variance of the median was found to be ```1.933``` and ```0.0166```, respectively.











