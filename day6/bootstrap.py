import numpy as np
import scipy.stats as stats

# Exercise 1

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


# Exercise 2


def var(X, mean):

    sum_var = 0

    for i in range(len(X)):
        sum_var += (X[i] - mean)**2

    return sum_var/(len(X)-1)


X = [5, 4, 9, 6, 21, 17, 11, 20, 7, 10, 21, 15, 13, 16, 8]

var_list = []
for i in range(100):

    pick = np.random.choice(X, size=15)

    var_list.append(var(pick, np.mean(X)))

print('Exercise 2')
print(np.var(var_list))


# EX 3

def get_samples(num_samples):
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
