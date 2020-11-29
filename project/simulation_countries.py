import numpy as np
from shapely.geometry import box
from population import Population
from person import Person
from countries import Countries
import os
from collections import defaultdict
import matplotlib.pyplot as plt


min_x = 0
min_y = 0
max_x = 100
max_y = 100

# Defines the size that the people can move within
# Box(min_x, min_y, max_x, max_y)
shape1 = box(min_x, min_y, 200, 200)
shape2 = box(min_x, min_y, max_x, max_y)
shape3 = box(min_x, min_y, 150, 150)
shape4 = box(min_x, min_y, max_x, max_y)
shape5 = box(min_x, min_y, max_x, max_y)
shape6 = box(min_x, min_y, 400, 400)
shape7 = box(min_x, min_y, max_x, max_y)
shape8 = box(min_x, min_y, 250, 250)
shape9 = box(min_x, min_y, max_x, max_y)


# Dictionary containing different shapes of each country
shape_dict = {
    '0': shape1,
    '1': shape2,
    '2': shape3,
    '3': shape4,
    '4': shape5,
    '5': shape6,
    '6': shape7,
    '7': shape8,
    '8': shape9
}

# Dictionary for population size
numpersons_dict = {
    '0': 500,
    '1': 200,
    '2': 200,
    '3': 600,
    '4': 100,
    '5': 1500,
    '6': 350,
    '7': 400,
    '8': 20
}

# Contains the % of death in each time step for every age group
death_rates = {
    '0': 0,
    '1': 0.001,
    '2': 0.005,
    '3': 0.008,
    '4': 0.01,
    '5': 0.01,
    '6': 0.04,
    '7': 0.04,
    '8': 0.06,
    '9': 0.06
}

# Dictionary containing how different contries move
popmove_dict = {
    '0': 2,
    '1': 1,
    '2': 10,
    '3': 2,
    '4': 2,
    '5': 4,
    '6': 2,
    '7': 2,
    '8': 2
}

# Number of populations
np.random.seed(42)
num_populations = 9

# Dictionary for people in each country
persons_dict = defaultdict(list)

# Initialize people for each population
for j in range(num_populations):

    # Find out how many persons to add to country
    for i in range(numpersons_dict[str(j)]):

        bounds = shape_dict[str(j)].bounds

        position = np.array(
            [np.random.uniform(bounds[0], bounds[2]), np.random.uniform(bounds[1], bounds[3])])

        person = Person(position, death_rates,
                        movement=popmove_dict[str(j)], mean_recovery=80)

       # START IN CHINA
        if j == 5 and (i % 200) == 0:
            person.infected = True
            person.recovery_time = 180

        # Add person to dictionary
        persons_dict[str(j)].append(person)

# Create populations
ES = Population(persons_dict['0'], shape1, hospital_factor=0.1, hospital_beds=200,
                quarantine_rate=0.001, mean_quarantine=2, name='ES', travel_time=5)

DE = Population(persons_dict['1'], shape2, hospital_factor=0.1, hospital_beds=100,
                quarantine_rate=0.02, mean_quarantine=2, name='Germany', travel_time=5)

SE = Population(persons_dict['2'], shape3, hospital_factor=0.2, hospital_beds=100,
                quarantine_rate=0.01, mean_quarantine=2, name='Sweden', travel_time=5)

FR = Population(persons_dict['3'], shape4, hospital_factor=0.1, hospital_beds=100,
                quarantine_rate=0.02, mean_quarantine=5, name='France', travel_time=5)

DK = Population(persons_dict['4'], shape5, hospital_factor=0.1, hospital_beds=100,
                quarantine_rate=0.02, mean_quarantine=5, name='Denmark', travel_time=5)

CH = Population(persons_dict['5'], shape6, hospital_factor=0.1, hospital_beds=100,
                quarantine_rate=0.05, mean_quarantine=10, name='China', travel_time=5)

IT = Population(persons_dict['6'], shape7, hospital_factor=0.01, hospital_beds=60,
                quarantine_rate=0, mean_quarantine=2, name='Italy', travel_time=5)

BE = Population(persons_dict['7'], shape8, hospital_factor=0.1, hospital_beds=100,
                quarantine_rate=0, mean_quarantine=2, name='Belgium', travel_time=5)

EST = Population(persons_dict['8'], shape9, hospital_factor=0.05, hospital_beds=1000,
                 quarantine_rate=0, mean_quarantine=2, name='Estonia', travel_time=5)

cont = Countries([ES, DE, SE, FR, DK, CH, IT, BE, EST], travels=True)

# Plot interactions
#steps = 600
cont.interact(steps=800, plot=True, save=False)


for population in cont.populations:
    plt.figure()

    plt.bar(list(population.dead_age.keys()),
            population.dead_age.values(), color='g')
    plt.show()

    plt.figure()

    # Show Simple Moving Average (MA) of infection rate
    ma_num = 7

    ma = np.convolve(population.infectionrate,
                     np.ones((ma_num,))/ma_num)[(ma_num-1):]

    while len(ma) < len(population.infectionrate):
        ma = np.insert(ma, 0, 0)

    plt.plot(population.infectionrate, c='b', label='Rate')
    plt.plot(ma, c='r', linestyle='dashed', label=f'{ma_num}_MA')
    plt.legend()
    plt.savefig(f'7_ma_{population.name}.png')
    plt.show()
