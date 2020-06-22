import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from collections import defaultdict
import random
plt.style.use('ggplot')

# Linear Congruential Generator


def lcg(a, c, m, init, total_numbers=10000, classes=10, plot=False):
    """
    Uses the Linear Congruential Generator to generate (pseudo) random uniform points
    """

    # Initialize list of numbers
    rand_numbers = [0] * total_numbers

    # Let first number be the initial value (given by user)
    rand_numbers[0] = init

    for i in range(1, total_numbers):
        # take modulus of numbers
        rand_numbers[i] = (a * rand_numbers[i-1] + c) % m

    # Divide by m
    rand_numbers = list(map(lambda x: x/m, rand_numbers))

    if plot:

        plt.hist(rand_numbers, bins=classes, rwidth=0.75)
        plt.plot([0, 1], [total_numbers/classes]*2, color='blue',
                 linestyle='dashed', label='Expected')
        plt.title(f'LCG: Histogram of {classes} classes')
        plt.legend()
        plt.show()

        plt.scatter(rand_numbers[:len(rand_numbers)-1], rand_numbers[1:])
        plt.title('LCG: Scatter plot of (numbers[1:-1], numbers[2:])')
        plt.show()
    return rand_numbers


def chi_squared(rand_numbers, num_classes=10):
    """
    Chi-Squared Test for uniform numbers
    """

    # Contains the frequency for each class
    dict_classes = defaultdict(int)

    # Create intervals that define classes in uniform numbers
    intervals = []
    for i in range(num_classes):
        intervals.append((i+1)/num_classes)

    for num in rand_numbers:
        for i in range(len(intervals)):
            if num > intervals[i]:
                continue
            else:
                dict_classes[i] += 1
                break

    total_length = len(rand_numbers)

    # loop over all classes and compute sum for test_stat
    test_stat = 0
    for i in range(num_classes):
        # sum for all the classes
        test_stat += ((dict_classes[i] - total_length /
                       num_classes)**2) / (total_length/num_classes)

    p_val = 1-stats.chi2.cdf(test_stat, num_classes-1)

    return test_stat, p_val


def kolmogorov(rand_numbers, granularity=1000, plot=False):
    """
    Kolmogorov test for uniform numbers
    """

    # Sort numbers
    sorted_numbers = sorted(rand_numbers)

    # Initialize lists

    # Empirical CDF
    f_x = [0] * granularity

    # Analytical CDF
    Fn_x = [0] * granularity

    # Compute analytical CDF
    for i in range(granularity):
        count = 0
        for num in sorted_numbers:
            if i/granularity >= num:
                count += 1
            else:
                break

        # Normalize cdf, such that it sums to 1
        f_x[i] = count/len(sorted_numbers)

        # Analyticl CDF
        Fn_x[i] = i/granularity

    # Find index with maximal difference
    diff = np.abs(np.array(f_x) - np.array(Fn_x))
    max_diff_idx = np.argmax(diff)
    max_diff = diff[max_diff_idx]

    test_stat = np.sqrt(len(rand_numbers) + 0.12 + 0.11 *
                        np.sqrt(len(rand_numbers))) * max_diff

    if plot:
        plt.plot(f_x, c='b', label='Empirical CDF')
        plt.plot(Fn_x, c='r', label='Analytical CDF')
        plt.plot(
            [max_diff_idx, max_diff_idx],
            [0, 1],
            color='green',
            linestyle='dashed',
            label=f'Suprema {round(max_diff, 6)}'
        )
        plt.legend()
        plt.title('LCG: Empirical vs. Analytical CDF Kolmogorov')
        plt.show()

    return test_stat


def run_test2(rand_numbers):
    amatrix = np.array([
        [4529.4, 9044.9, 13568, 18091, 22615, 27892],
        [9044.9, 18097, 27139, 36187, 45234, 55789],
        [13568, 27139, 40721, 54281, 67852, 83685],
        [18091, 36187, 54281, 72414, 90470, 111580],
        [22615, 45234, 67852, 90470, 113262, 139476],
        [27892, 55789, 83685, 111580, 139476, 172860]])

    barray = np.array([1/6, 5/24, 11/120, 19/720,
                       29/5040, 1/840]).reshape(-1, 1)

    run_vector = np.zeros(6)
    idx = 0

    run_counter = -1
    last_val = -1
    while idx < len(rand_numbers):
        # print(run_counter)
        if rand_numbers[idx] >= last_val:
            run_counter += 1
            last_val = rand_numbers[idx]

            idx += 1

        else:
            if run_counter >= 5:
                run_vector[5] = run_vector[5] + 1
            else:
                run_vector[run_counter] = run_vector[run_counter] + 1

            last_val = -1
            run_counter = -1

    if run_counter >= 5:
        run_vector[5] = run_vector[5] + 1
    else:
        run_vector[run_counter] = run_vector[run_counter] + 1

    n = len(rand_numbers)

    run_vector = run_vector.reshape(-1, 1)

    step1 = (run_vector - n * barray).T @ amatrix

    step2 = step1 @ (run_vector - n*barray)

    Z = 1/(n-6) * step2

    return Z


"""vals = [0.54, 0.67, 0.13, 0.89, 0.33, 0.45,
        0.90, 0.01, 0.45, 0.76, 0.82, 0.24, 0.17]

print(run_test2(vals))"""


def run_test(rand_numbers):
    """
    Performs run test on random uniform numbers
    """

    run_lengths = []

    condition_order = []
    for i in range(len(rand_numbers) - 1):
        if rand_numbers[i] > rand_numbers[i+1]:
            condition_order.append('>')
        else:
            condition_order.append('<')

    current = condition_order[0]
    count = 0
    for i in range(len(condition_order)):
        if condition_order[i] == current:
            count += 1
        else:
            current = condition_order[i]
            run_lengths.append(count)
            count = 1

    test_stat = (len(run_lengths) - (2*len(rand_numbers) - 1)/3) / \
        np.sqrt((16*len(rand_numbers)-29)/90)

    return test_stat


def correlation_test(rand_numbers, h):
    sum_val = 0
    for i in range(len(rand_numbers)-h):
        sum_val = rand_numbers[i] * rand_numbers[i+h]

    corr = 1/(len(rand_numbers)-h) * sum_val

    return corr


print('#'*10)
print('PERFORMING TEST ON LCG RNG:')

rand_numbers = lcg(a=4629, c=17, m=65536, init=2353,
                   total_numbers=10000, plot=True)

chi_sq_stat, chi_sq_pval = chi_squared(rand_numbers, num_classes=10)

print(f'Chi Squared Test Statistics: {chi_sq_stat}')
print(f'Chi Squared P-value: {chi_sq_pval}')

ks_stat = kolmogorov(rand_numbers, plot=True)

print(f'Kolmogorov-smirnov Test Statistics: {round(ks_stat, 6)}')


run_test_stat = run_test(rand_numbers)
print(f'Run Test III Test Statistic: {run_test_stat}')

print(f'N(0, 1) 95% critical value: {stats.norm.ppf(0.95)}')

run_test2_stat = run_test2(rand_numbers)
print(f'run_testII {run_test2_stat}')

corr_test = correlation_test(rand_numbers, 5)
print(f'Correlation test using h=5: {corr_test}')

# PERFORM TEST ON PYTHON RNG
print('#'*10)
print('PERFORMING TEST ON PYTHON RNG')

rand_numbers_python = [random.random() for i in range(10000)]

plt.hist(rand_numbers_python, bins=10, rwidth=0.75)
plt.plot([0, 1], [1000]*2, color='blue', linestyle='dashed', label='Expected')
plt.title('PYTHON: Histogram of 10 classes')
plt.legend()
plt.show()


plt.scatter(rand_numbers_python[:len(
    rand_numbers_python)-1], rand_numbers_python[1:])
plt.title('PYTHON: Scatter plot of (numbers[1:-1], numbers[2:])')
plt.show()

chi_sq_stat, chi_sq_pval = chi_squared(rand_numbers_python, num_classes=10)

print(f'Chi Squared Test Statistics: {chi_sq_stat}')
print(f'Chi Squared P-value: {chi_sq_pval}')

ks_stat = kolmogorov(rand_numbers_python, plot=True)

print(f'Kolmogorov-smirnov Test Statistics: {round(ks_stat, 6)}')

run_test_stat = run_test(rand_numbers_python)
print(f'Run Test III Test Statistic: {run_test_stat}')

corr_test = correlation_test(rand_numbers_python, 5)
print(f'Correlation test using h=5: {corr_test}')
