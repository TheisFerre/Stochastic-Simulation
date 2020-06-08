import random
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def exp_dist(lamb, num_samples=10000, plot=False):

    samples = [0] * num_samples
    for i in range(num_samples):
        rand = random.random()

        samples[i] = -math.log(rand)/lamb

    if plot:
        plt.plot(list(range(num_samples)), sorted(samples, reverse=True))
        plt.title('Exponential Distribution')
        plt.show()

    return samples


def normal_dist(num_samples=10000, plot=False):

    samples = []
    while len(samples) < num_samples:
        u1 = random.random()

        v1 = random.uniform(-1, 1)
        v2 = random.uniform(-1, 1)

        r_squared = v1**2 + v2**2

        # got to next iteration
        if r_squared > 1:
            continue

        else:
            z1 = math.sqrt(-2*math.log(u1)) * v1/math.sqrt(r_squared)

            z2 = math.sqrt(-2*math.log(u1)) * v2/math.sqrt(r_squared)

            samples.append(z1)
            samples.append(z2)

    if plot:
        plt.hist(samples, bins=100)
        plt.title('Sampling from normal distribution')
        plt.show()

    return samples


def pareto(beta, k, num_samples=10000, plot=False, moments=False):

    samples = [0] * num_samples

    for i in range(num_samples):

        rand = random.random()

        samples[i] = beta * (rand**(-1/k))

    if plot:
        plt.plot(list(range(num_samples)), sorted(samples, reverse=True))
        plt.title(f'Pareto, beta={beta}, k={k}')
        plt.show()

    if moments:
        empirical_mean = np.array(samples).mean()
        empirical_var = np.array(samples).var()

        return samples, empirical_mean, empirical_var

    return samples


samples_exp = exp_dist(3, plot=True)
test_stat_exp, critical_vals, significance = stats.anderson(
    samples_exp, dist='expon')

print(f'Exponential Test statistic: {test_stat_exp}')
print(f'Exponential Critical values: {critical_vals}')
print(f'Exponential Significance levels: {significance}')


samples_norm = normal_dist(plot=True)
test_stat_norm, p_val_norm = stats.kstest(samples_norm, 'norm')

print(f'Normal Test statistic: {test_stat_norm}')
print(f'Normal p-value: {p_val_norm}')


# Pareto
beta = 1
for k in [2.05, 2.5, 3, 4]:
    _, emp_mean, emp_var = pareto(beta=beta, k=k, plot=True, moments=True)

    analytical_mean = k/(k-1) * beta
    analytical_var = k / ((k-1)**2 * (k-2)) * beta**2

    print(f'For k={k}')
    print(
        f'Absolute difference emp. vs analytical mean: {abs(emp_mean - analytical_mean)}')
    print(
        f'Absolute difference emp. vs analytical variance: {abs(emp_var - analytical_var)}')


# Create 100 95% confidence intervals
# from normal distribution based on 10 samples
num_samples = 10
mean_list = []
upper_list = []
lower_list = []
for i in range(100):
    samples = np.array(normal_dist(num_samples=num_samples))

    mean = samples.mean()
    std = samples.std()

    conf_val = std/np.sqrt(num_samples) * stats.t.ppf(1-0.05/2, num_samples-1)

    mean_list.append(mean)

    upper = mean + conf_val
    lower = mean - conf_val

    upper_list.append(upper)
    lower_list.append(lower)

    #plt.plot([i, i], [lower, upper], c='b')

#plt.plot(list(range(100)), mean_list, c='r')
#plt.title('100 95% Confidence intervals')
# plt.show()

# Plot 100 Confidence intervals
plt.plot(list(range(100)), upper_list, c='b', linestyle='dashed')
plt.plot(list(range(100)), lower_list, c='b', linestyle='dashed')
plt.plot(list(range(100)), mean_list, c='r')
plt.title('100 normal dist. 95% Confidence Intervals')
plt.show()
