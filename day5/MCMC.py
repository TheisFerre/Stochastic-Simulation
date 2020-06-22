import math
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare
from collections import defaultdict, Counter


class MCMC():
    def __init__(self, func, draw_func, position, proposal_dist=False):

        self.func = func
        self.proposal_dist = proposal_dist
        self.draw_func = draw_func

        # Represent our current position or "x"
        self.position = position

    def sample(self):
        """
        Take a new sample "y" and see if we should reject
        """
        rand = random.random()

        y = self.draw_func()

        # Create a flag that checks if we are within boundaries
        # If we are not, we will not use the sample!
        flag = False

        if type(y) == int:
            if self.position + y == - 1 or self.position+y == 10:
                flag = True
            if not flag:
                accept_prob = min(1, self.func(
                    self.position + y)/self.func(self.position))

        else:
            y = tuple([self.position[i] + y[i]
                       for i in range(len(self.position))])

            if sum(y) >= 10:
                flag = True

            if y[0] == -1:
                flag = True

            if y[1] == -1:
                flag = True

            if not flag:
                accept_prob = min(1, self.func(*y)/self.func(*self.position))

        # Accept
        if flag:
            return self.position

        if rand < accept_prob:
            if type(y) == int:
                self.position = self.position + y
            else:
                self.position = y

        return self.position

    def run(self, num_samples, chains=1, burn_period=0):

        samples = []

        while len(samples) < num_samples:

            if len(samples) == 0:
                samples.append(self.sample())
                continue

            sample = self.sample()
            if samples[-1] == sample:
                continue
            else:
                samples.append(sample)

        return samples


n = 10
# EX 1


def func1(x, alpha=3, n=10):

    nominator = alpha**x / math.factorial(x)

    denom = sum([alpha**j / math.factorial(j) for j in range(n)])

    return nominator/denom


def draw_func1():
    rand = random.random()
    if rand > 0.5:
        return 1
    else:
        return -1


def count_samples(samples):
    # Contains the frequency for each class
    dict_classes = defaultdict(int)

    for num in samples:
        dict_classes[num] += 1

    max_val = max(dict_classes, key=dict_classes.get)
    print(max_val)

    class_list = [0] * max_val

    for i in range(max_val+1):
        try:
            class_list[i] = dict_classes[i]
        except:
            pass

    return class_list


model = MCMC(func=func1, draw_func=draw_func1, position=5)

samples = model.run(100)

sample_hist = count_samples(samples)
print('CHI SQUARED EXERCISE 1')
print(chisquare(f_obs=sample_hist))

plt.hist(samples, rwidth=0.85)
plt.title('MCMC 1-variable')
plt.show()


# EX2
n = 10


def func2(*args, alpha1=4, alpha2=4):

    return alpha1**args[0] / math.factorial(args[0]) * alpha2**args[1] / math.factorial(args[1])


def draw_func2():
    rand1 = random.random()
    rand2 = random.random()

    if rand1 > 0.5:
        y1 = 1
    else:
        y1 = -1

    if rand2 > 0.5:
        y2 = 1
    else:
        y2 = -1

    return (y1, y2)


func_draw = draw_func2

model = MCMC(func=func2, draw_func=func_draw, position=(4, 4))
samples = model.run(num_samples=500)


x_vals = [tup[0] for tup in samples]
y_vals = [tup[1] for tup in samples]

x_samples = count_samples(x_vals)
y_samples = count_samples(y_vals)

print('CHI SQUARED EXERCISE 2')
print(chisquare(f_obs=x_samples))
print(chisquare(f_obs=y_samples))


plt.plot(x_vals, y_vals)
plt.title('MCMC 2-variables')
plt.show()


barWidth = 0.25

x_bins = np.bincount(x_vals)
y_bins = np.bincount(y_vals)

r1 = np.arange(len(x_bins))
r2 = [x + barWidth for x in r1]

plt.bar(r1, x_bins, width=barWidth)
plt.bar(r2, y_bins, width=barWidth)
plt.title('MCMC(X,Y)-variables alpha1=4, alpha2=4')
plt.show()


# EX3


alpha1 = 17
alpha2 = 12
n = 40

# Marginalize functions


def p_i_given_j(i, j, alpha1=alpha1, alpha2=alpha2, n=n):

    nomin = ((alpha1**i)/math.factorial(i) * (alpha2**j)/math.factorial(j))

    denom = sum([alpha1**i / math.factorial(i)
                 for i in range(n-j)])

    return nomin/(denom * alpha2**j/math.factorial(j))


def p_j_given_i(i, j, alpha1=alpha1, alpha2=alpha2, n=n):

    nomin = ((alpha1**i)/math.factorial(i) * (alpha2**j)/math.factorial(j))

    denom = sum([alpha2**j / math.factorial(j)
                 for j in range(n-i)])

    return nomin/(denom * alpha1**i/math.factorial(i))


class MCMC_gibs():
    def __init__(self, func1, func2, num_classes, position, proposal_dist=False):

        self.p_i_given_j = func1
        self.p_j_given_i = func2
        self.proposal_dist = proposal_dist
        self.start_cond = 0
        self.num_classes = num_classes

        # Represent our current position or "x"
        self.position = position

    def sample(self):
        """
        Sample with gibbs sampling
        """

        # Create a flag that checks if we are within boundaries
        # If we are not, we will not use the sample!

        if self.start_cond == 0:

            j = self.position[1]

            probs = [self.p_i_given_j(i, j) for i in range(self.num_classes-j)]
            int_choice = np.random.choice(list(range(len(probs))), p=probs)

            self.position = (int_choice, self.position[1])
            self.start_cond = 1

        else:

            i = self.position[0]

            probs = [self.p_j_given_i(i, j) for j in range(self.num_classes-i)]
            int_choice = np.random.choice(list(range(len(probs))), p=probs)

            self.position = (self.position[0], int_choice)
            self.start_cond = 0

        return self.position

    def run(self, num_samples, chains=1, burn_period=0):

        samples = []

        while len(samples) < num_samples:

            if len(samples) == 0:
                samples.append(self.sample())
                continue

            sample = self.sample()
            samples.append(sample)

        return samples


model = MCMC_gibs(func1=p_i_given_j, func2=p_j_given_i,
                  num_classes=n, position=(int(n/2)-1, int(n/2)-1))
samples = model.run(num_samples=1000)

x_vals = [tup[0] for tup in samples]
y_vals = [tup[1] for tup in samples]

x_samples = count_samples(x_vals)
y_samples = count_samples(y_vals)

print('CHI SQUARED EXERCISE 3')
print(chisquare(f_obs=x_samples))
print(chisquare(f_obs=y_samples))


plt.plot(x_vals, y_vals)
plt.title(f'Gibbs Sampling, {n}-classes, ({alpha1},{alpha2})-Alpha values')
plt.show()

c_x = Counter()
c_y = Counter()

for i in range(len(x_vals)):
    c_x[x_vals[i]] += 1
    c_y[y_vals[i]] += 1


x_list = [0] * n
y_list = [0] * n
for i in range(n):
    x_list[i] = c_x[i]
    y_list[i] = c_y[i]

print(x_list)
print(y_list)


barWidth = 0.25

r1 = np.arange(len(x_list))
r2 = [x + barWidth for x in r1]

plt.bar(r1, x_list, width=barWidth)
plt.bar(r2, y_list, width=barWidth)
plt.title('GIBBS(X,Y)-variables alpha1=17, alpha2=12')
plt.show()
