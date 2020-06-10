from scipy.stats import poisson, erlang, pareto, bernoulli, expon, t
from math import sqrt, exp, log
import numpy as np
import matplotlib.pyplot as plt
import random


class ServiceUnit():
    """
    Represents a Service Unit
    """

    def __init__(self, mean_service_time, last_service=0, service_available=True, service_time_dist='expon'):

        # self.last_service = last_service
        self.next_service = 0

        self.service_available = service_available
        self.mean_service_time = mean_service_time
        self.service_time_dist = service_time_dist

    def exp_dist(self):
        rand = random.random()
        time_step = -self.mean_service_time * log(rand)

        return time_step

    def sample_service_time(self, time):
        """
        Sample next availability for service unit
        """

        if self.service_time_dist == 'expon':
            service_avail_time = time + self.exp_dist()

        elif self.service_time_dist == 'pareto':
            service_avail_time = time + pareto.rvs(self.mean_service_time)

        self.next_service = service_avail_time

    def update_service_status(self, time):
        """
        Function for updating service status of unit
        """
        if time >= self.next_service:
            self.service_available = True


class BlockSystem():
    """
    Class object to simulate a Discrete Event simulation program for a blocking system
    with no waiting room
    """

    def __init__(self,
                 num_service_units,
                 mean_service_time,
                 mean_customer_arrival,
                 customer_arrival_dist='expon',
                 service_time_dist='expon',
                 ):

        self.num_service_units = num_service_units
        self.mean_service_time = mean_service_time
        self.mean_customer_arrival = mean_customer_arrival
        self.customer_arrival_dist = customer_arrival_dist
        self.service_time_dist = service_time_dist

        # If we needed a waiting room
        # self.queue = deque()

    def exp_dist(self):
        rand = random.random()
        time_step = -self.mean_customer_arrival * log(rand)

        return time_step

    def customer_sample(self):
        """
        Function to draw bernoulli sample from Poisson dist.
        Used for arrival
        """

        if self.customer_arrival_dist == 'expon':
            #time_step = poisson.rvs(self.mean_customer_arrival)
            time_step = self.exp_dist()

        elif self.customer_arrival_dist == 'erlang':
            time_step = erlang.rvs(self.mean_customer_arrival)

        elif self.customer_arrival_dist == 'hyperexp':
            rand1 = random.random()
            rand2 = random.random()

            if rand1 < self.mean_customer_arrival[0][0]:
                time_step = -1/self.mean_customer_arrival[0][1] * log(rand2)
            else:
                time_step = -1/self.mean_customer_arrival[1][1] * log(rand2)

        return time_step

    def simulate(self, simulation_customers=10000, plot=False):
        """
        Simulate system
        """

        num_blocked = 0
        num_serviced = 0

        # Initialize lists for blocked/serviced customers
        service_available_list = [0] * (simulation_customers + 1)
        blocked_list = [0] * (simulation_customers + 1)
        serviced_list = [0] * (simulation_customers + 1)
        time_steps = [0]

        # Initialise service units
        service_units = [ServiceUnit(self.mean_service_time, service_time_dist=self.service_time_dist)
                         for i in range(self.num_service_units)]

        for service_unit in service_units:
            service_unit.last_service = 0
            service_unit.service_available = True

        iter_count = 1

        # Simulate system
        while num_blocked + num_serviced < simulation_customers:

            time_step = self.customer_sample()
            time_steps.append(time_steps[-1] + time_step)

            service_avail_count = 0

            # Update service units:
            for service_unit in service_units:
                service_unit.update_service_status(time_steps[-1])

                # Count number of service units available
                if service_unit.service_available:
                    service_avail_count += 1

            # Service the customer
            customer_serviced = False
            for service_unit in service_units:
                if service_unit.service_available:
                    customer_serviced = True

                    service_unit.service_available = False
                    service_unit.sample_service_time(time_steps[-1])
                    break

            # Log customer service
            if customer_serviced:
                num_serviced += 1
            else:
                num_blocked += 1

            # Log blocked and serviced
            blocked_list[iter_count] = num_blocked
            serviced_list[iter_count] = num_serviced
            service_available_list[iter_count] = service_avail_count

            iter_count += 1

        # Plot evolution of blocked vs. serviced units
        if plot:
            plt.plot(time_steps,
                     blocked_list, color='b', label='Blocked')

            plt.plot(time_steps,
                     serviced_list, color='r', label='serviced')

            plt.plot(time_steps, service_available_list,
                     color='g', label='Service Units Available')

            plt.text(len(blocked_list)-1,
                     blocked_list[-1], str(blocked_list[-1]))
            plt.text(len(serviced_list)-1,
                     serviced_list[-1], str(serviced_list[-1]))

            plt.title(
                f'#Service Units: {self.num_service_units}, Service Time: {self.mean_service_time}, Customer Arrival: {self.mean_customer_arrival}')
            plt.legend()
            plt.show()

        return blocked_list, serviced_list, service_available_list

    def __call__(self, simulations=10, simulation_customers=10000, alpha=0.05, plot=False):
        """
        Simulate system multiple times
        """

        simulations_list = []

        # Create a number of different simulations
        for _ in range(simulations):
            simulations_list.append(self.simulate(simulation_customers))

        simulations_mat = np.array(simulations_list)

        total_customers = simulations_mat[:, 0, 1:] + simulations_mat[:, 1, 1:]

        # Compute statistics
        means = (simulations_mat[:, 0, 1:] /
                 total_customers).mean(axis=0)
        std = (simulations_mat[:, 0, 1:] /
               total_customers).std(axis=0)
        CI_arr = std / sqrt(simulations) * t.ppf(1-alpha/2, simulations-1)

        conf_neg = means - CI_arr
        conf_pos = means + CI_arr

        # Plot confidence intervals
        if plot:
            plt.plot(list(range(simulation_customers)),
                     means, color='r', label='Mean Fraction')

            plt.plot(list(range(simulation_customers)), conf_pos, color='r',
                     linestyle='dashed')
            plt.plot(list(range(simulation_customers)), conf_neg, color='r',
                     linestyle='dashed', label=f'{1-alpha}% Confidence interval')

            plt.text(simulation_customers-1,
                     means[-1], str(round(means[-1], 2)))

            plt.title(
                f'Final Fraction: {round(means[-1], 3)}, Confidence: {round(means[-1], 3)}+-{round(CI_arr[-1], 3)}')

            plt.legend()
            plt.show()
        return means, CI_arr


# TESTING
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

blocked, serviced, service_avail = system.simulate(
    simulation_customers=10000, plot=True)

system(simulations=10, simulation_customers=10000, plot=True)

# Pareto
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

# Hyper Exponential
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
