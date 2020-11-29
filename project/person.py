import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import bernoulli, poisson
from shapely.geometry import Point


class Person():

    def __init__(self, position, death_rate, movement=5, mean_recovery=140, return_time=5):
        """
        Position: list of (X, Y)-coordinates
        death_rate: Dictionary of deathrates for different age groups
        movement: Parameter for controlling the movement of a person
        mean_recovery: Parameter for mean recovery of person
        return_time: Parameter for how long time a person visits another country
        """

        # Variable for age
        self.age = math.floor(np.random.randint(low=1, high=100)/10)
        self.death_rate = death_rate

        # position is a np array of (x, y) coordinate
        self.position = position

        self.origin = self
        self.return_time = return_time

        self.movement = movement

        self.infected = False
        self.numinfected = 0

        # Mean recovery rate of 140 hours?
        self.recovery_time = None
        self.mean_recovery = mean_recovery
        self.dead = False
        self.immune = False

        # Quarantine Variables
        self.quarantined = False
        self.quarantine_period = None

    def move(self, shape):
        """
        Function for moving a person by a random walk
        Changes position of this person
        """

        move = np.random.normal(scale=self.movement, size=2)
        new_cord = Point(self.position + move)

        # Reject if not in shape
        while not shape.contains(new_cord):
            move = np.random.normal(scale=self.movement, size=2)
            new_cord = Point(self.position + move)

        self.position = self.position + move

    def dist(self, person):
        """
        Function for computing distance between itself and another person
        """
        distance = math.sqrt(
            (self.position[0] - person.position[0])**2 + (self.position[1] - person.position[1])**2)
        return distance

    def infect(self, persons):
        """
        Function for infecting other people
        """
        newly_infected = set()
        for person in persons:
            # Check if person is all ready infected
            if person.infected:
                continue
            elif person.immune:
                continue
            elif person.quarantined:
                continue
            else:
                # Check if distance to person is less than or equal to 2 (meters?)
                # Infect if true
                if self.dist(person) <= 2:
                    distance = 2-self.dist(person)
                    p = bernoulli.rvs(distance*0.5)
                    if p == 1:
                        person.infected = True
                        newly_infected.add(person)
                        self.numinfected += 1

        return newly_infected

    def update_infected(self, step):
        """
        Function for updating status of person
        """
        # Way of defining draw for recovery time for a person
        recovery_time = poisson.rvs(self.mean_recovery)
        self.recovery_time = step + recovery_time

        return None

    def update_recovery(self, step):
        """
        Function for updating status of person
        """
        # Way of updating the recovery time - when the recovery time counts down to the
        # original step, then the person is immune and cannot be infected
        if self.recovery_time == step:
            # making them  immune
            self.immune = True
            self.infected = False

        return None

    def update_death(self, hospital_availability=True, hospital_factor=0.8):
        # Hospital availability. If the number of infected people is lower
        # than 10, death rate decreases.
        if hospital_availability:
            p = bernoulli.rvs(self.death_rate[str(self.age)]*hospital_factor)
            if p == 1:
                self.dead = True
                return True
        else:
            p = bernoulli.rvs(self.death_rate[str(self.age)])
            if p == 1:
                self.dead = True
                return True

        return False

    def update_quarantine(self, step, rate, mean_quarantine):

        p = bernoulli.rvs(rate)

        if p == 1:
            self.quarantined = True

            quarantine_time = mean_quarantine
            self.quarantine_period = step + quarantine_time

            return True
        else:
            return False
