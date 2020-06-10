import random
import numpy as np
import math
import time
import scipy.stats as stats


# EX1
def monte_carlo_est(num_samples=100):

    estimations = [0] * num_samples

    for i in range(num_samples):
        rand = random.random()
        estimations[i] = math.exp(rand)

    std = np.std(estimations)
    CI = std/math.sqrt(num_samples) * stats.t.ppf(0.05, num_samples-1)

    return np.mean(estimations), CI

# EX2


def antithetic_est(num_samples=100):
    estimations = [0] * num_samples

    for i in range(num_samples):
        rand = random.random()
        exp_num = math.exp(rand)

        estimations[i] = (exp_num + math.exp(1)/exp_num) / 2

    std = np.std(estimations)
    CI = std/math.sqrt(num_samples) * stats.t.ppf(0.05, num_samples-1)

    return np.mean(estimations), CI


def control_est(num_samples=100):
    estimations = [0] * num_samples

    c = -1.69056

    for i in range(num_samples):
        rand = random.random()

        estimations[i] = math.exp(rand) + c*(rand - 1/2)

    std = np.std(estimations)
    CI = std/math.sqrt(num_samples) * stats.t.ppf(0.05/2, num_samples-1)

    return np.mean(estimations), CI


def strat_est(num_samples=100, strata=10):

    estimations = [0] * num_samples

    for i in range(num_samples):
        # sample number, divide by strata and use expon
        rand_nums = np.exp(np.random.random(10)/strata)

        # Compute expon values e^(0/10), e^(1/10)...e^(9/10)
        exp_vals = np.exp(np.arange(0, 1, 1/strata))

        sample = np.sum(rand_nums * exp_vals) / 10

        estimations[i] = sample

    std = np.std(estimations)
    CI = std/math.sqrt(num_samples) * stats.t.ppf(0.05/2, num_samples-1)

    return np.mean(estimations), CI


print(f'True Evaluation of integral: {math.exp(1) - 1}')

# TEST EX1
print('Monte Carlo estimation')
time_montecarlo = time.time()
est_montecarlo, CI_montecarlo = monte_carlo_est(num_samples=100)
print(f'Estimation: {est_montecarlo}')
print(f'Confidence Interval: {est_montecarlo}+-{CI_montecarlo}')
print(f'finish time: {time.time() - time_montecarlo}')
print('#'*5)

# TEST EX2
print('Anithetic Variables estimation')
time_antithetic = time.time()
est_antithetic, CI_antithetic = antithetic_est(num_samples=100)
print(f'Estimation: {est_antithetic}')
print(f'Confidence Interval: {est_antithetic}+-{CI_antithetic}')
print(f'finish time: {time.time() - time_antithetic}')
print('#'*5)


# TEST EX3
print('Control Variate estimation')
time_contr = time.time()
est_contr, CI_contr = control_est(num_samples=100)
print(f'Estimation: {est_contr}')
print(f'Confidence Interval: {est_contr}+-{CI_contr}')
print(f'finish time: {time.time() - time_contr}')
print('#'*5)


# TEST EX4
print('Stratified Sampling estimation')
time_strat = time.time()
est_strat, CI_strat = strat_est(num_samples=100)
print(f'Estimation: {est_strat}')
print(f'Confidence Interval: {est_strat}+-{CI_strat}')
print(f'finish time: {time.time() - time_strat}')
print('#'*5)
