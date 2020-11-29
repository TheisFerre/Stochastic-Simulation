from celluloid import Camera
from person import Person
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict


class Population():
    def __init__(self, persons, shape, hospital_beds=100, hospital_factor=0.8, quarantine_rate=0.02, mean_quarantine=20, name=None, travel_time=5, lockdown_threshold=None, lockdown_time=None):
        """
        persons: list of people in population
        shape: Shape of country
        hospital_beds: Parameter for controlling the critical value for hospitals in country
        hospital_factor: Parameter for controlling the factor of the death rate. (low factor=lower death rate)
        quarantine_rate: Parameter for controlling the rate of infected that get quarantined
        mean_quarantine: The period that a person is quarantined
        name: Sets the name of the population/country
        travel_time: Parameter for the travel time
        """

        self.name = name

        # persons is a list
        self.persons = persons

        # Make it a set for speed up (order does not matter!)
        # self.persons = set()
        # for _ in range(num_persons):
        #   self.persons.add(Person(2, 2))

        #(minx, miny, maxx, maxy)
        self.shape = shape

        self.travel_time = travel_time

        self.visitors = []

        self.lockdown_threshold = lockdown_threshold
        self.lockdown_time = lockdown_time
        self.lockdown = False

        self.infected = []
        for person in self.persons:
            if person.infected:
                self.infected.append(person)

        self.immune = []
        # Make list of all infected people
        # Make it a set for speed up (order does not matter!)
        # self.infected = set()
        # for person in self.persons:
        #   if person.infected:
        #       self.infected.add(person)

        # Count number of deadpeople
        self.dead = 0

        self.num_quarantined = 0

        # Quarantine Rate
        self.quarantine_rate = quarantine_rate
        self.mean_quarantine = mean_quarantine

        # Death count on age
        self.dead_age = defaultdict(int)

        # Calculation for infection rates
        self.infectionrate = []

        # Hospital parameters
        self.hospital_beds = hospital_beds
        self.hospital_factor = hospital_factor

    def run(self, steps, plot=False, test=False):

        if test:
            healthy_list = [len(self.persons) - len(self.infected)]
            sick_list = [len(self.infected)]
            dead_list = [self.dead]
            immune_list = [len(self.immune)]

        if plot:
            fig1, ax1 = plt.subplots()
            camera1 = Camera(fig1)

            x_vals_healthy = [person.position[0] for person in self.persons]
            y_vals_healthy = [person.position[1] for person in self.persons]

            ax1.scatter(x_vals_healthy, y_vals_healthy, c='g')

            x_vals_sick = [person.position[0] for person in self.infected]
            y_vals_sick = [person.position[1] for person in self.infected]

            ax1.scatter(x_vals_sick, y_vals_sick, c='r')

            x_vals_immune = [person.position[0] for person in self.immune]
            y_vals_immune = [person.position[1] for person in self.immune]

            plt.scatter(x_vals_immune, y_vals_immune, c='b')

            camera1.snap()

            fig2, ax2 = plt.subplots()
            camera2 = Camera(fig2)

            healthy_list = [len(self.persons) - len(self.infected)]
            sick_list = [len(self.infected)]
            dead_list = [self.dead]
            immune_list = [len(self.immune)]

            ax2.plot([0], healthy_list, c='g', label='Healthy')
            ax2.plot([0], sick_list, c='r', label='Infected')
            ax2.plot([0], dead_list, c='black', label='Dead')
            ax2.plot([0], immune_list, c='b', label='Immune')
            ax2.legend()
            # ax2.plot(quarantinecount)
            # ax2.plot(self.dead)
            # ax2.plot(resistantcount)
            camera2.snap()

        for step in range(steps):

            for person in self.persons:
                # Move every person
                person.move(self.shape)

                # Check if someone is quarantined
                if person.quarantined:
                    if person.quarantine_period == step:
                        person.quarantined = False
                        if person.recovery_time <= step:
                            person.immune = True
                        else:
                            self.infected.append(person)

            print(len(self.infected))

            recovered = list()

            for infected in self.infected:

                newly_infected = infected.infect(self.persons)

                infected.update_recovery(step)
                if infected.immune:
                    recovered.append(infected)
                    self.immune.append(infected)
                # Go through newly infected people and add if they are not in infected list
                for new in newly_infected:
                    # We update the person an gives a random recovery time when infected
                    new.update_infected(step)

                    if new not in self.infected:
                        self.infected.append(new)

                if len(self.infected) <= self.hospital_beds:
                    hospital_availability = True
                else:
                    hospital_availability = False

                if infected.update_death(hospital_availability, self.hospital_factor):
                    self.infected.remove(infected)
                    self.persons.remove(infected)
                    self.dead += 1
                    self.dead_age[infected.age] += 1

                elif infected.update_quarantine(step, self.quarantine_rate, self.mean_quarantine):
                    self.infected.remove(infected)

            for person in recovered:
                if person in self.infected:
                    self.infected.remove(person)

            sum_infected = 0

            for infected in self.infected:
                sum_infected += infected.numinfected * infected.mean_recovery
                infected.numinfected = 0

            if len(self.infected) != 0:
                mean_infected = sum_infected / len(self.infected)
                self.infectionrate.append(mean_infected)

            else:
                self.infectionrate.append(0)

            if test:
                healthy_list.append(len(self.persons) - len(self.infected))
                sick_list.append(len(self.infected))
                dead_list.append(self.dead)
                immune_list.append(len(self.immune))

            if plot:
                x_vals_healthy = [person.position[0]
                                  for person in self.persons]
                y_vals_healthy = [person.position[1]
                                  for person in self.persons]

                ax1.scatter(x_vals_healthy, y_vals_healthy, c='g')

                x_vals_sick = [person.position[0] for person in self.infected]
                y_vals_sick = [person.position[1] for person in self.infected]

                ax1.scatter(x_vals_sick, y_vals_sick, c='r')

                x_vals_immune = [person.position[0] for person in self.immune]
                y_vals_immune = [person.position[1] for person in self.immune]

                ax1.scatter(x_vals_immune, y_vals_immune, c='b')

                camera1.snap()

                healthy_list.append(len(self.persons) - len(self.infected))
                sick_list.append(len(self.infected))
                dead_list.append(self.dead)
                immune_list.append(len(self.immune))

                ax2.plot(list(range(step+2)), healthy_list, c='g')
                ax2.plot(list(range(step+2)), sick_list, c='r')
                ax2.plot(list(range(step+2)), dead_list, c='black')
                ax2.plot(list(range(step+2)), immune_list, c='b')

                camera2.snap()

        if plot:
            animation1 = camera1.animate(blit=True)
            animation1.save('movement.mp4')

            animation2 = camera2.animate(blit=False)
            animation2.save('evolution.mp4')

        if test:
            plt.figure()
            plt.plot(list(range(step+2)), healthy_list, c='g', label='Healthy')
            plt.plot(list(range(step+2)), sick_list, c='r', label='Infected')
            plt.plot(list(range(step+2)), dead_list, c='black', label='Dead')
            plt.plot(list(range(step+2)), immune_list, c='b', label='Immune')
            plt.legend()
            plt.show()

        print(self.dead)
