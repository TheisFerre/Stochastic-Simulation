from scipy.stats import bernoulli, expon, t
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
        self.services = []

    def exp_dist(self):
        rand = random.random()
        time_step = - self.mean_service_time * log(rand)

        return time_step

    def sample_service_time(self, time):

        service_avail_time = time + self.exp_dist()

        if len(self.services) == 0:
            self.services.append(service_avail_time)
        else:
            self.services.append(service_avail_time - time)

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

    def exp_dist(self, rand_num):
        rand = rand_num
        time_step = -self.mean_customer_arrival * log(rand)

        return time_step

    def customer_sample(self, rand_num):
        """
        Sample next time for customer
        """

        def exp_dist(self, rand_num):
            rand = rand_num
            time_step = -self.mean_customer_arrival * log(rand)

            return time_step

        if self.customer_arrival_dist == 'expon':
            # time_step = poisson.rvs(self.mean_customer_arrival)
            time_step = self.exp_dist(rand_num)

        elif self.customer_arrival_dist == 'hyperexp':
            rand1 = rand_num

            if rand1 >= self.mean_customer_arrival[0][0]:
                time_step = -1/self.mean_customer_arrival[0][1] * log(1-rand1)
            else:
                time_step = -1/self.mean_customer_arrival[1][1] * log(1-rand1)

        return time_step

    def simulate(self, rand_numbers, plot=False):
        """
        Simulate the Block system using random numbers
        """

        num_blocked = 0
        num_serviced = 0

        # Initialize lists for blocked/serviced customers
        service_available_list = [0] * (len(rand_numbers) + 1)
        blocked_list = [0] * (len(rand_numbers) + 1)
        serviced_list = [0] * (len(rand_numbers) + 1)
        time_steps = [0]

        # Initialise service units
        service_units = [ServiceUnit(self.mean_service_time)
                         for i in range(self.num_service_units)]

        for service_unit in service_units:
            service_unit.last_service = 0
            service_unit.service_available = True

        iter_count = 1

        # Simulate system
        while num_blocked + num_serviced < len(rand_numbers):

            time_step = self.customer_sample(rand_numbers[iter_count-1])

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

        mean_service_time = []
        for service_unit in service_units:
            mean_service_time.append(np.mean(service_unit.services))

        mean_service_time = np.mean(mean_service_time)

        return blocked_list, serviced_list, service_available_list, mean_service_time

    def __call__(self,  rand_numbers, simulations=10, alpha=0.05, plot=False):
        """
        Run multiple simulations and plot mean/confidence interval
        """

        simulations_list = []
        mean_services = []

        # Create a number of different simulations
        for i in range(simulations):
            blocked_list, serviced_list, service_available_list, mean_service_time = self.simulate(
                rand_numbers[i])

            mean_services.append(mean_service_time)
            simulations_list.append(
                (blocked_list, serviced_list, service_available_list))

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
            plt.plot(list(range(len(means))),
                     means, color='r', label='Mean Fraction')

            plt.plot(list(range(len(conf_pos))), conf_pos, color='r',
                     linestyle='dashed')
            plt.plot(list(range(len(conf_neg))), conf_neg, color='r',
                     linestyle='dashed', label=f'{1-alpha}% Confidence interval')

            plt.text(len(means)-1,
                     means[-1], str(round(means[-1], 2)))

            plt.title(
                f'Final Fraction: {round(means[-1], 3)}, Confidence: {round(means[-1], 3)}+-{round(CI_arr[-1], 3)}')

            plt.legend()
            plt.show()
        return means, CI_arr


# COMMON RANDOM NUMBER TESTING:
# Hyper Exponential Arrival times

num_service_units_hyper = 10
mean_service_time_hyper = 1
mean_customer_arrival_hyper = [(0.8, 0.8333), (0.2, 5)]

system_hyper = BlockSystem(
    num_service_units=num_service_units_hyper,
    mean_service_time=mean_service_time_hyper,
    mean_customer_arrival=mean_customer_arrival_hyper,
    customer_arrival_dist='hyperexp',
    service_time_dist='expon'
)

# Exponential Arrival times
num_service_units_exp = 10
mean_service_time_exp = 8
mean_customer_arrival_exp = 1

system_exp = BlockSystem(
    num_service_units=num_service_units_exp,
    mean_service_time=mean_service_time_exp,
    mean_customer_arrival=mean_customer_arrival_exp,
    customer_arrival_dist='expon',  # expon/erlang/hyperexp
    service_time_dist='expon'  # expon/pareto
)


blocked_frac_exp = []
blocked_frac_hyper = []

num_simulations = 20
num_random = 10000

for i in range(num_simulations):
    rand_numbers = np.random.rand(num_random)

    blocked_exp, _, _, _ = system_exp.simulate(rand_numbers)
    blocked_frac_exp.append(blocked_exp[-1]/num_random)

    blocked_hyper, _, _, _ = system_hyper.simulate(rand_numbers)
    blocked_frac_hyper.append(blocked_hyper[-1]/num_random)

estimator_list = np.array(blocked_frac_exp - np.array(blocked_frac_hyper))

mean_estimator = estimator_list.mean()

std_estimator = estimator_list.std()

CI = std_estimator/np.sqrt(num_simulations) * \
    t.ppf(1-0.05/2, num_simulations-1)

print(f'MEAN OF ESTIMATOR: {mean_estimator}')

print(f'CI: +-{CI}')
