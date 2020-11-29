from population import Population
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle


class Countries:
    def __init__(self, populations, travels=False):
        """
        populations: list of populations
        travels: Parameter for setting travels true/false
        """

        self.populations = populations
        self.travels = travels

    def step_population(self, step, population):

        # Check for population lockdown threshold.
        # Only a single lockdown can be enforced in each country.
        if population.lockdown_threshold is not None:
            if len(population.infected)/len(population.persons) >= population.lockdown_threshold:
                population.lockdown = True
                population.lockdown_time = population.lockdown_time + step
                # Only a single lockdown
                population.lockdown_threshold = None

        if population.lockdown:
            if step == population.lockdown_time:
                population.lockdown = False

        # Random walk for all people in population including visitors
        for person in population.persons + population.visitors:
            if not population.lockdown:
                person.move(population.shape)

            # Check if someone is quarantined
            if person.quarantined:
                if person.quarantine_period == step:
                    person.quarantined = False
                    population.num_quarantined -= 1

                    if person.recovery_time <= step:
                        person.immune = True
                    else:
                        population.infected.append(person)

        # Initialize empty list for recovered people
        recovered = list()

        # Iterate through infected people including visitors
        infected_visitors = [
            visitor for visitor in population.visitors if visitor.infected is True]
        for infected in population.infected + infected_visitors:

            newly_infected = infected.infect(
                population.persons+population.visitors)

            # Go through newly infected people and add if they are not in infected list
            for new in newly_infected:
                # We update the person an gives a random recovery time when infected
                new.update_infected(step)

                if new not in population.infected and new not in population.visitors:
                    population.infected.append(new)

            # Update to immune if they recovered
            infected.update_recovery(step)
            if infected.immune:
                recovered.append(infected)

                # If infected person is not a visitor add to immune and infected
                if infected not in population.visitors:
                    population.immune.append(infected)
                    population.infected.remove(infected)
                continue

            # Check if hospital beds are available
            if len(population.infected) <= population.hospital_beds:
                hospital_availability = True
            else:
                hospital_availability = False

            # Check if person dies. Only infected people can die
            if infected.update_death(hospital_availability, population.hospital_factor):
                if infected in population.infected:
                    population.infected.remove(infected)

                if infected in population.persons:
                    population.persons.remove(infected)

                if infected.quarantined:
                    population.num_quarantined -= 1

                if infected in population.visitors:
                    population.visitors.remove(infected)

                population.dead += 1
                population.dead_age[infected.age] += 1

            # Check if person is out of quarantine
            elif infected.update_quarantine(step, population.quarantine_rate, population.mean_quarantine):
                population.num_quarantined += 1
                if infected in population.infected:
                    population.infected.remove(infected)

        # If person has recovered. Remove from infected list
        for person in recovered:
            if person in population.infected:
                population.infected.remove(person)

    def interact(self, steps, plot=False, save=False):
        """
        Main loop for interactions between countries
        """

        if plot:
            plot_dict = dict()

            for population in self.populations:
                plot_dict[str(population)] = {
                    'Susceptible': [len(population.persons) - len(population.infected) - len(population.immune)],
                    'Infected': [len(population.infected)],
                    'Dead': [population.dead],
                    'Immune': [len(population.immune)]
                }
                if save:

                    x_vals_healthy = [person.position[0]
                                      for person in population.persons]
                    y_vals_healthy = [person.position[1]
                                      for person in population.persons]

                    x_vals_sick = [person.position[0]
                                   for person in population.infected]
                    y_vals_sick = [person.position[1]
                                   for person in population.infected]

                    x_vals_immune = [person.position[0]
                                     for person in population.immune]
                    y_vals_immune = [person.position[1]
                                     for person in population.immune]

                    movement_dict = {
                        'Susceptible': [[x_vals_healthy, y_vals_healthy]],
                        'Infected': [[x_vals_sick, y_vals_sick]],
                        'Immune': [[x_vals_immune, y_vals_immune]]
                    }

        for step in range(steps):
            print(f'step :{step}')
            visitor_idx = []
            new_visitor = []

            # Pick a random traveller from each population to move to another population
            if self.travels:
                for population in self.populations:
                    # print(len(population.persons))
                    if len(population.persons) > 0:
                        visitor_idx.append(np.random.randint(
                            0, len(population.persons)))

                        person = population.persons.pop(visitor_idx[-1])

                        if person in population.infected:
                            population.infected.remove(person)

                        if person in population.immune:
                            population.immune.remove(person)

                        person.origin = population
                        person.return_time = step + population.travel_time

                        new_visitor.append(person)

                # Choose a population to move to that is not the same
                for i in range(len(visitor_idx)):
                    # Choose a population
                    choices = list(range(len(self.populations)))
                    choices.pop(i)
                    choice = np.random.choice(choices)

                    # Update position
                    bounds = self.populations[choice].shape.bounds

                    new_visitor[i].position = np.array([np.random.uniform(
                        bounds[0], bounds[2]), np.random.uniform(bounds[1], bounds[3])])

                    # print(new_visitor[i].return_time)

                    self.populations[choice].visitors.append(new_visitor[i])

            for population in self.populations:
                # print(step)
                self.step_population(step, population)

            # Check if visitor is about to return back to origin
            for population in self.populations:

                for visitor in population.visitors:
                    if int(visitor.return_time) == int(step):
                        population.visitors.remove(visitor)

                        if visitor.infected:
                            visitor.origin.infected.append(visitor)
                            visitor.origin.persons.append(visitor)
                        elif visitor.immune:
                            visitor.origin.immune.append(visitor)
                            visitor.origin.persons.append(visitor)
                        else:
                            visitor.origin.persons.append(visitor)

                        # Modify the position of the person
                        bounds = visitor.origin.shape.bounds

                        visitor.position = np.array([np.random.uniform(
                            bounds[0], bounds[2]), np.random.uniform(bounds[1], bounds[3])])

            # Calculate infectionrate from current time step
            for population in self.populations:

                sum_infected = 0

                infected_visitors = [
                    visitor for visitor in population.visitors if visitor.infected is True]

                for infected in population.infected + infected_visitors:

                    sum_infected += infected.numinfected * infected.mean_recovery
                    infected.numinfected = 0

                if len(population.infected) != 0:
                    mean_infected = sum_infected / len(population.infected)
                    population.infectionrate.append(mean_infected)
                else:
                    population.infectionrate.append(0)

            # Plot infection chart for each population
            if plot:
                for population in self.populations:

                    visitor_susc = 0
                    visitor_inf = 0
                    visitor_im = 0

                    for visitor in population.visitors:
                        if visitor.infected:
                            visitor_inf += 1
                        elif visitor.immune:
                            visitor_im += 1
                        else:
                            visitor_susc += 1

                    plot_dict[str(population)]['Susceptible'].append(visitor_susc +
                                                                     len(population.persons) - len(population.infected) - len(population.immune))
                    plot_dict[str(population)]['Infected'].append(visitor_inf +
                                                                  len(population.infected))
                    plot_dict[str(population)]['Dead'].append(population.dead)
                    plot_dict[str(population)]['Immune'].append(visitor_im +
                                                                len(population.immune))

                    # Save data for each population
                    if save:
                        x_vals_healthy = [person.position[0]
                                          for person in population.persons]
                        y_vals_healthy = [person.position[1]
                                          for person in population.persons]

                        x_vals_sick = [person.position[0]
                                       for person in population.infected]
                        y_vals_sick = [person.position[1]
                                       for person in population.infected]

                        x_vals_immune = [person.position[0]
                                         for person in population.immune]
                        y_vals_immune = [person.position[1]
                                         for person in population.immune]

                        movement_dict['Susceptible'].append(
                            [x_vals_healthy, y_vals_healthy])
                        movement_dict['Infected'].append(
                            [x_vals_sick, y_vals_sick])
                        movement_dict['Immune'].append(
                            [x_vals_immune, y_vals_immune])

        # Infection chart plotting
        if plot:
            suscep = np.zeros((len(range(step+2))))
            Infec = np.zeros(len(range(step+2)))
            Imm = np.zeros(len(range(step+2)))
            Dea = np.zeros(len(range(step+2)))

            for idx, population in enumerate(self.populations):

                plt.figure()
                plt.plot(list(
                    range(step+2)), plot_dict[str(population)]['Susceptible'], c='g', label='Susceptible')

                plt.plot(
                    list(range(step+2)), plot_dict[str(population)]['Infected'], c='r', label='Infected')
                plt.plot(list(range(step+2)),
                         plot_dict[str(population)]['Dead'], c='black', label='Dead')
                plt.plot(list(range(step+2)),
                         plot_dict[str(population)]['Immune'], c='b', label='Immune')

                suscep = suscep + \
                    np.array(plot_dict[str(population)]['Susceptible'])
                Infec = Infec + \
                    np.array(plot_dict[str(population)]['Infected'])
                Imm = Imm + np.array(plot_dict[str(population)]['Immune'])
                Dea = Dea + np.array(plot_dict[str(population)]['Dead'])

                plt.legend()
                if population.name is not None:
                    plt.title(f'{population.name}-Infection Chart')
                    plt.savefig(f'InfectionChart_{population.name}.png')
                else:
                    plt.savefig(f'{str(idx)}_InfectionChart.png')

                plt.figure()
                plt.plot(list(range(step+2)), suscep,
                         c='g', label='Susceptible')
                plt.plot(list(range(step+2)), Infec, c='r', label='Infected')
                plt.plot(list(range(step+2)), Imm, c='b', label='Immune')
                plt.plot(list(range(step+2)), Dea, c='black', label='Dead')
                plt.title(f'World Simulation')
                plt.legend()
                plt.savefig('World_infectionchart.png')

                # Save data to pickle files
                if save:
                    print('Saving info to pickle files:')

                    with open('movement.pickle', 'wb') as handle:
                        pickle.dump(movement_dict, handle,
                                    protocol=pickle.HIGHEST_PROTOCOL)

                    with open('evolution.pickle', 'wb') as handle:
                        pickle.dump(plot_dict, handle,
                                    protocol=pickle.HIGHEST_PROTOCOL)
