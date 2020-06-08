from scipy.stats import poisson, erlang, pareto, bernoulli, expon, t
from math import sqrt, exp
import numpy as np
import matplotlib.pyplot as plt


class ServiceUnit():
    """
    Represents a Service Unit
    """

    def __init__(self, mean_service_time, last_service=0, service_available=True, service_time_dist='expon'):

        self.last_service = last_service
        self.service_available = service_available
        self.mean_service_time = mean_service_time
        self.service_time_dist = service_time_dist

    def update_service_status(self):
        """
        Function for updating service status of unit
        """

        if self.service_time_dist == 'expon':
            prob = expon.cdf(self.last_service, self.mean_service_time)
            draw = int(bernoulli.rvs(prob))
        elif self.service_time_dist == 'pareto':
            prob = pareto.cdf(self.last_service, self.mean_service_time)
            draw = int(bernoulli.rvs(prob))

        if draw:
            self.service_available = True
            self.last_service = 0
        else:
            self.last_service += 1


class BlockSystem():
    """
    Class object to simulate a Discrete Event simulation program for a blocking system
    with no waiting room
    """

    def __init__(self,
                 num_service_units,
                 mean_service_time,
                 mean_customer_arrival,
                 customer_arrival_dist='poisson',
                 service_time_dist='expon',
                 ):

        self.num_service_units = num_service_units
        self.mean_service_time = mean_service_time
        self.mean_customer_arrival = mean_customer_arrival
        self.customer_arrival_dist = customer_arrival_dist
        self.service_time_dist = service_time_dist

        # If we needed a waiting room
        # self.queue = deque()

    def binary_customer_sample(self, x):
        """
        Function to draw bernoulli sample from Poisson dist.
        Used for arrival
        """
        if self.customer_arrival_dist == 'poisson':
            prob = poisson.cdf(x, self.mean_customer_arrival)
            draw = int(bernoulli.rvs(prob))

        elif self.customer_arrival_dist == 'erlang':
            prob = erlang.cdf(x, self.mean_customer_arrival)
            draw = int(bernoulli.rvs(prob))

        elif self.customer_arrival_dist == 'hyperexp':
            prob = 0
            for params in self.mean_customer_arrival:
                prob += params[0] * exp(-params[1] * x)
            draw = int(bernoulli.rvs(1-prob))

        else:
            prob = poisson.cdf(x, self.mean_customer_arrival)
            draw = int(bernoulli.rvs(prob))

        return draw

    def simulate(self, simulation_period=500, plot=False):

        num_blocked = 0
        num_serviced = 0

        # Initialize lists for blocked/serviced customers
        service_available_list = [0] * simulation_period
        blocked_list = [0] * simulation_period
        serviced_list = [0] * simulation_period

        last_customer_arrival = 0

        # Initialise service units
        service_units = [ServiceUnit(self.mean_service_time, service_time_dist=self.service_time_dist)
                         for i in range(self.num_service_units)]

        for service_unit in service_units:
            service_unit.last_service = 0
            service_unit.service_available = True

        iter_count = 0

        # Simulate system
        while simulation_period > iter_count:

            service_avail_count = 0
            # Update service units:
            for service_unit in service_units:
                service_unit.update_service_status()

                # Count number of service units available
                if service_unit.service_available:
                    service_avail_count += 1

            # Question for TA's, is it only a single customer that arrives, or multiple (10)?

            # See if customer arrives, assume customer arrives at timepoint 0:
            if self.binary_customer_sample(last_customer_arrival) or iter_count == 0:
                last_customer_arrival = 0
                customer_serviced = False

                # Service the customer
                for service_unit in service_units:
                    if service_unit.service_available:
                        customer_serviced = True

                        service_unit.service_available = False
                        service_unit.last_service = 0
                        break

                # Log customer service
                if customer_serviced:
                    num_serviced += 1
                else:
                    num_blocked += 1

            # Increment last arrival of customer
            else:
                last_customer_arrival += 1

            # Log blocked and serviced
            blocked_list[iter_count] = num_blocked
            serviced_list[iter_count] = num_serviced
            service_available_list[iter_count] = service_avail_count

            iter_count += 1

        # Plot evolution of blocked vs. serviced units
        if plot:
            plt.plot(list(range(len(blocked_list))),
                     blocked_list, color='b', label='Blocked')

            plt.plot(list(range(len(serviced_list))),
                     serviced_list, color='r', label='serviced')

            plt.plot(list(range(len(service_available_list))), service_available_list,
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

    def __call__(self, simulations=10, simulation_period=500, alpha=0.05, plot=False):

        simulations_list = []

        # Create a number of different simulations
        for i in range(simulations):
            simulations_list.append(self.simulate(simulation_period))

        # Initialize lists for statistics
        mean_fraction_list = [0] * simulation_period
        var_fraction_list = [0] * simulation_period
        conf_interval = [0] * simulation_period

        # Compute statistics
        for i in range(simulation_period):
            sum_frac = 0
            sum_frac_squared = 0
            for j in range(simulations):

                # Blocked/total
                fraction = simulations_list[j][0][i] / \
                    (simulations_list[j][1][i] + simulations_list[j][0][i])

                sum_frac += fraction
                sum_frac_squared += fraction**2

            mean_fraction_list[i] = sum_frac/simulations
            var_fraction_list[i] = 1/(simulations-1) * (sum_frac_squared -
                                                        simulations * mean_fraction_list[i]**2)

            conf_interval[i] = sqrt(
                var_fraction_list[i])/sqrt(simulations) * t.ppf(1-alpha/2, simulations-1)

        # Create +- confidence intervals on fractions
        conf_interval_neg = np.array(
            mean_fraction_list) - np.array(conf_interval)
        conf_interval_pos = np.array(
            mean_fraction_list) + np.array(conf_interval)

        # Plot confidence intervals
        if plot:
            plt.plot(list(range(simulation_period)),
                     mean_fraction_list, color='r', label='Mean Fraction')

            plt.plot(list(range(simulation_period)), conf_interval_pos, color='r',
                     linestyle='dashed')
            plt.plot(list(range(simulation_period)), conf_interval_neg, color='r',
                     linestyle='dashed', label=f'{1-alpha}% Confidence interval')

            plt.text(simulation_period-1,
                     mean_fraction_list[-1], str(
                         round(mean_fraction_list[-1], 2)))

            plt.title(
                f'Final Fraction: {round(mean_fraction_list[-1], 3)}, Confidence: {round(mean_fraction_list[-1], 3)}+-{round(conf_interval[-1], 3)}')

            plt.legend()
            plt.show()

        return self.simulate(simulation_period)


# TESTING
num_service_units = 10
mean_service_time = 35
mean_customer_arrival = 1

system = BlockSystem(
    num_service_units=num_service_units,
    mean_service_time=mean_service_time,
    mean_customer_arrival=mean_customer_arrival,
    customer_arrival_dist='erlang',
    service_time_dist='expon'
)


"""
mean_customer_arrival = [(0.8, 0.8333), (0.2, 5)]
# Hyper Exponential
system = BlockSystem(
    num_service_units=num_service_units,
    mean_service_time=mean_service_time,
    mean_customer_arrival=mean_customer_arrival,
    customer_arrival_dist='hyperexp'
)
"""

blocked, serviced, service_avail = system.simulate(
    simulation_period=1000, plot=True)

system(simulations=5, simulation_period=500, plot=True)
