import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt

# EX1


def dist(p1, p2):

    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def temp(k, mul=1):

    return 1/math.sqrt(1+k) * mul

    # return -math.log(k+1)


def energy(points):

    distance = 0
    for i in range(len(points)):

        if i == len(points) - 1:
            distance += dist(points[i], points[0])
        else:
            distance += dist(points[i], points[i+1])

    return distance


def func(x, k):

    return math.exp(-energy(x)/temp(k))


def permute_points(points):

    points_copy = points.copy()

    rand1 = random.randint(0, len(points)-1)
    rand2 = rand1

    while rand2 == rand1:
        rand2 = random.randint(0, len(points)-1)

    points_copy[rand1], points_copy[rand2] = points_copy[rand2], points_copy[rand1]

    return points_copy

# Change temp for every second step (maybe)


points = []

for num in np.linspace(0, 2*np.pi, 8):

    points.append([np.cos(num), np.sin(num)])

np.random.shuffle(points)

cost_start = round(energy(points), 3)

x_vals = [p[0] for p in points]
y_vals = [p[1] for p in points]

x_vals.append(x_vals[0])
y_vals.append(y_vals[0])

plt.plot(x_vals, y_vals, c='g')

samples = []
num_samples = 500
for i in range(num_samples):

    y = permute_points(points)

    if i % 10 == 0:
        k = (i+1)/10

    k = (i+1)

    if energy(y) < energy(points):
        points = y
    else:
        accept_prob = min(1, func(y, k) / func(points, k))

        rand1 = random.random()

        if rand1 < accept_prob:
            points = y

    samples.append(points)


x_vals = [p[0] for p in samples[-1]]
y_vals = [p[1] for p in samples[-1]]

cost_end = round(energy(samples[-1]), 3)

x_vals.append(x_vals[0])
y_vals.append(y_vals[0])

plt.plot(x_vals, y_vals)
plt.scatter(x_vals, y_vals, c='r', s=30)
plt.title(f'Start cost {cost_start}, End cost {cost_end}')
plt.show()


def energy_cost(stations, cost_mat):

    cost = 0
    for i in range(len(stations)):

        if i == len(stations) - 1:
            cost += cost_mat[stations[i], stations[0]]
        else:
            cost += cost_mat[stations[i], stations[i+1]]

    return cost


def func2(x, k, cost_mat):

    return math.exp(- (energy_cost(x, cost_mat)) / temp(k))


df = pd.read_csv('cost.csv', header=None)
costs = df.values/1000

stations = np.array(list(range(len(costs))))

np.random.shuffle(stations)

cost_list = []
cost_start = round(energy_cost(stations, cost_mat=costs), 3)

cost_list.append(cost_start)

samples = []
num_samples = 50000
for i in range(num_samples):

    y = permute_points(stations)

    if i % 10 == 0:
        k = (i+1)/10

    if energy_cost(y, costs) < energy_cost(stations, costs):
        stations = y

    else:

        fy = func2(y, k, costs)
        fx = func2(stations, k, costs)

        accept_prob = min(1, fy / fx)

        rand1 = random.random()

        if rand1 < accept_prob:
            stations = y

    samples.append(stations)

    cost_list.append(round(energy_cost(stations, cost_mat=costs), 3))


cost_end = round(energy_cost(samples[-1], cost_mat=costs), 3)

print(samples[-1])

# print(cost_start)
# print(min(cost_list))

plt.plot(cost_list)
plt.title(f'start cost: {cost_start}, end cost: {cost_end}')
plt.ylabel('cost')
plt.show()
