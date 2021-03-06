import random
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats

plt.style.use('ggplot')


def direct_method(probs, samples=10000, plot=False):
    """
    Function for direct sampling method
    """

    sample_values = []

    for i in range(samples):
        sample = random.random()

        for j in range(len(probs)):
            if sample > sum(probs[0:j+1]):
                continue
            else:
                sample_values.append(j+1)
                break

    if plot:
        plt.hist(sample_values, bins=len(probs), rwidth=0.75)
        plt.title('Direct sample method')
        plt.show()

    return sample_values


def rejection_method(probs, sample_values=10000, plot=False):
    """
    Function for rejection sampling method
    """

    # Compute c
    k = len(probs)
    c = max(k * probs)

    num_reject = 0
    samples = []

    while len(samples) < sample_values:
        # get U1, U2
        rand1 = random.random()
        rand2 = random.random()

        # Get indicator variable
        indicator = math.floor(k * rand1) + 1

        # see if we should reject
        if rand2 <= probs[indicator-1]/c:
            samples.append(indicator)

        # Reject and try once more
        else:
            num_reject += 1
            continue

    if plot:
        plt.hist(samples, bins=k, rwidth=0.75)
        plt.title(
            f'Rejection method, {num_reject} rejected')
        plt.show()

    return samples


def alias_method(probs, num_samples=10000, plot=False):
    """
    Function for alias sampling method
    """

    # Generate F and L
    k = len(probs)

    L = list(range(1, k+1))
    F = k * probs

    # Generate g and s
    g = list(np.array(L)[(F >= 1)])
    s = list(np.array(L)[(F <= 1)])

    while len(s) > 0:

        i, j = g[0], s[0]

        L[j-1] = i
        F[i-1] = F[i-1] - (1-F[j-1])

        if F[i-1] < 1 - 1e-5:

            # Remove first element
            g.pop(0)

            # Add i to S
            s.append(i)

        s.pop(0)

    samples = []

    while len(samples) < num_samples:
        rand1 = random.random()
        rand2 = random.random()

        indicator = math.floor(k * rand1) + 1

        if rand2 <= F[indicator-1]:
            samples.append(indicator)
        else:
            samples.append(L[indicator-1])

    if plot:
        plt.hist(samples, bins=len(probs), rwidth=0.75)
        plt.title('Alias Method')
    plt.show()

    return samples


# Probabilities that will be used for simulation
probs = np.array([7/48, 5/48, 1/8, 1/16, 1/4, 5/16])

plt.bar(list(range(1, len(probs) + 1)), probs)
plt.title('Probabilites')
plt.show()

num_samples = 10000

# Direct method
samples_direct = direct_method(probs, samples=num_samples, plot=True)
frequencies_direct = np.bincount(samples_direct)[1:]

chisq_direct, pval_direct = stats.chisquare(
    frequencies_direct, probs*num_samples)
print(f'Direct Test statistic {chisq_direct}')
print(f'Direct pval {pval_direct}')

# Rejection method
samples_reject = rejection_method(probs, sample_values=num_samples, plot=True)
frequencies_reject = np.bincount(samples_reject)[1:]

chisq_reject, pval_reject = stats.chisquare(
    frequencies_reject, probs*num_samples)
print(f'Reject Test statistic {chisq_reject}')
print(f'Reject pval {pval_reject}')


# Alias method
samples_alias = alias_method(probs, num_samples=num_samples, plot=True)
frequencies_alias = np.bincount(samples_alias)[1:]

chisq_alias, pval_alias = stats.chisquare(frequencies_alias, probs*num_samples)
print(f'Alias Test statistic {chisq_alias}')
print(f'Alias pval {pval_alias}')
