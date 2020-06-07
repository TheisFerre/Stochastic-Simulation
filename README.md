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

Test for correlation
```python
plt.scatter(numbers[:len(numbers)-1], numbers[1:])
```
![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day1/Scatter_plot_corr.png)

Chi-squared Test
```python
Test_statistic, p_val = chi_squared(numbers, num_classes=10)
```

```
Test_statistic = 5.986
p_val = 0.7413
```

Kolmogorov-Smirnov Test
```python
Adjusted_test_statistic = kolmogorov(numbers)
```
```
Adjusted_test_statistic = 1.02
```
![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day1/Kolmogorov.png)

Run-Test III
```python
Test_statistic = run_test(numbers)
```
```
Test_statistic = 0.869
```

## Day 3

This day we was supposed to simulate "Block system". This system contains ```n``` service units, mean service unit time, mean customer arrival.
The system simulates customers that show up to a store to get a service from one of the ```n``` service units. The Service units mean availability is modelled with an exponential function. Customer arrival mean is modeled as a Poisson process.

```python
num_service_units = 10
mean_service_time = 45
mean_customer_arrival = 3

system = BlockSystem(
    num_service_units=num_service_units,
    mean_service_time=mean_service_time,
    mean_customer_arrival=mean_customer_arrival
)
```
System Dynamics:

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day3/Block_system_simulation.png)

95% Confidence interval on fraction blocked/serviced

![Screenshot](https://github.com/TheisFerre/Stochastic-Simulation/blob/master/day3/Block_system_simulation_confidence.png)




