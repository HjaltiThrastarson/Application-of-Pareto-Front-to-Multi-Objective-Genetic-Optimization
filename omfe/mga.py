# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


from logging import exception
import numpy as np
from numpy.typing import NDArray
import sys
from .problems import ChankongHaimes


class MicroGeneticAlgorithm:
    def __init__(
        self,
        fitness_functions,
        constraint_functions,
        num_variables,
        population_size=50,
        agents_to_keep=10,
        agents_to_shuffle=8,
        random_restarts=10,
        max_iterations=20,
        num_bits=8,
        random_seed=42,
    ):

        ## User input class variables

        # List of functions which have input of agent with dimensoality of 1xnum_variables
        self.functions = fitness_functions
        # List of constraint functions similar to fitness functions
        self.constraint_functions = constraint_functions
        # Number of variables i.e dimension of agent
        self.num_variables = num_variables
        # How many random agents to generate
        self.population_size = population_size
        # How many agents to keep after each iteration
        self.agents_to_keep = agents_to_keep
        # How many numbers to shuffle after each iteration
        self.agents_to_shuffle = agents_to_shuffle

        if self.agents_to_keep < self.agents_to_shuffle:
            raise ValueError("agents_to_keep must be smaller than agents_to_shuffle")
        if self.agents_to_keep > self.population_size:
            raise ValueError(
                "Number of agents to keep cannot be greater than population size"
            )

        # How many random restarts to do
        self.random_restarts = random_restarts
        # How many iterations per random restart to do
        self.max_iterations = max_iterations
        # Number of bits to use in gray_mapping, more --> slower but more accurate
        self.num_bits = num_bits
        # Set random seed for generating agents
        np.random.seed(random_seed)

        ## Internal class variables

        # Function weights
        self.weights = get_random_vec_with_sum_one(length=len(self.functions))

        # Current Fitness of all agents and corresponding variables
        self.fitness = np.zeros(population_size)
        # Fitness history, so that we can plot it
        self.fitness_history = np.zeros(
            (random_restarts, max_iterations, population_size)
        )
        # Best fitness so far
        self.best_fitness = np.ones((random_restarts)) * np.inf
        # Initial random start for agents
        self.agents = self.initialize_agents()
        # Agent history
        self.agents_history = np.zeros(
            (random_restarts, max_iterations, population_size, num_variables)
        )
        self.best_agents = np.zeros((random_restarts, num_variables))

    def initialize_agents(self):
        ## Random restart for agents
        agents = np.zeros((self.population_size, self.num_variables))
        for i in range(self.population_size):
            for j in range(self.num_variables):
                # Generates random variables in range -20 to 20
                constraints_broken = True
                while constraints_broken:
                    agents[i][j] = (np.random.rand(1) - 0.5) * 40
                    for k in range(len(self.constraint_functions)):
                        if self.constraint_functions[k](agents[i]) == 1:
                            constraints_broken = True
                        else:
                            constraints_broken = False
        return agents

    def calculate_fitness_function(self, random_restart, iteration):
        ## Calculates fitness for all agents with equal weighting for all functions
        fitness = 0
        for i in range(len(self.agents)):
            for j in range(len(self.functions)):
                # Calculate fitness for each function and add them up
                fitness += self.functions[j](self.agents[i]) * self.weights[j]
            for k in range(len(self.constraint_functions)):
                # Calculate fitness for each constraint and add them up
                if self.constraint_functions[k](self.agents[i]) == 1:
                    fitness += 1000000
            self.fitness[i] = fitness
            if fitness < self.best_fitness[random_restart]:
                self.best_fitness[random_restart] = fitness
                self.best_agents[random_restart][:] = self.agents[i]
            self.fitness_history[random_restart][iteration][i] = fitness
            self.agents_history[random_restart][iteration][i] = self.agents[i]

    def evaluate_agents(self):
        ## Evalutes fitness of each agent and decides which to keep
        # This is a magic line that sorts the agents by fitness
        agents_sorted = np.array([x for _, x in sorted(zip(self.fitness, self.agents))])
        return agents_sorted[: self.agents_to_keep]

    def shuffle_agents(self, agents_to_keep):
        ## Shuffle agents, generate a random cutoff point and
        ## select the best self.number_to_shuffle agents to be trailing agents
        new_agents = np.zeros((self.population_size, self.num_variables))
        agents_to_keep_fully = self.agents_to_keep - self.agents_to_shuffle
        new_agents[:agents_to_keep_fully] = agents_to_keep[:agents_to_keep_fully]
        # Generate random cutoff point
        cutoff = np.random.randint(0, self.num_bits)
        # Set first agent to keep fully to trailing agent
        trailing_agents = 1
        for i in range(agents_to_keep_fully, self.agents_to_keep):
            for j in range(self.num_variables):
                # Don't know how to do without for loop, might be a better way
                for h in range(agents_to_keep_fully):
                    if trailing_agents == h:
                        if h == trailing_agents:
                            trailing_agents += 1
                            # Map trailing and leading to graycode
                            gray_code_mapping_leading = self.gray_mapping(
                                agents_to_keep[i][j]
                            )
                            grat_code_mapping_trailing = self.gray_mapping(
                                agents_to_keep[trailing_agents - 1][j]
                            )
                            # Combine them to make a new agent
                            new_agent = (
                                gray_code_mapping_leading[:cutoff]
                                + grat_code_mapping_trailing[cutoff:]
                            )
                            new_agent = self.reversegray_mapping(int(new_agent, 2))
                            new_agents[i][j] = new_agent
                            break
                        if trailing_agents == agents_to_keep_fully:
                            trailing_agents = 1

        # Randomly generate the rest of the agents
        for i in range(self.agents_to_keep, self.population_size):
            for j in range(self.num_variables):
                new_agents[i][j] = (np.random.rand(1) - 0.5) * 40
        self.agents = new_agents

    def gray_mapping(self, number):
        ## gray_mapping of agents
        # These ranges should be variable specific, can later be implemented in the class
        min_range = -20
        max_range = 20
        total_range = abs(max_range - min_range)
        intervals = total_range / (2**self.num_bits - 1)
        # Make sure number is non-negative
        normalizedNumber = number + abs(min_range)
        # Calculate the number of intervals the number is in
        numberOfIntervals = int(normalizedNumber / intervals)
        # Convert to graycode
        binaryNumber = numberOfIntervals ^ (numberOfIntervals >> 1)
        return format(binaryNumber, "0{}b".format(self.num_bits))

    def reversegray_mapping(self, number):
        ## Does reverse gray_mapping, takes in a number
        # Takes in gray code decimal equailant number and converts to correct binary number equivalent
        # Taking xor until
        # n becomes zero
        inv = 0
        while number:
            inv = inv ^ number
            number = number >> 1
        number = inv
        min_range = -20
        max_range = 20
        total_range = abs(max_range - min_range)
        intervals = total_range / (2**self.num_bits - 1)
        number = number * intervals
        number = number - abs(min_range)
        return number

    def run_iterations(self):
        for random_restart in range(self.random_restarts):
            for iteration in range(self.max_iterations):
                self.calculate_fitness_function(random_restart, iteration)
                agents_to_keep = self.evaluate_agents()
                self.shuffle_agents(agents_to_keep)
            print(
                f"Random restart: {random_restart} starting, best fitness is {self.best_fitness[random_restart]} \
            and best agent is {self.best_agents[random_restart]}"
            )
            print(self.weights)
            self.weights = get_random_vec_with_sum_one(length=len(self.functions))


def get_random_vec_with_sum_one(length: int) -> NDArray[np.float64]:
    """Generate a vector that sums up to 1 with uniform distribution

    This can be imagined as cutting a string at random locations and measuring
    the distance of the resulting pieces, thus the ends 0 and 1 need to be added.

    Note: This is NOT choosing every coordinate randomly and then rescaling
    to [0,1], as this would not result in a uniform distribution. See
    https://stackoverflow.com/a/8068956 for an explanation attempt.
    """
    cuts = np.concatenate([0, np.random.random(size=length - 1), 1], axis=None)
    cuts.sort()
    return np.diff(cuts)


def main():
    problem = ChankongHaimes()
    fitness_functions = [problem.f_1, problem.f_2]
    constraint_functions = [problem.g_1, problem.g_2]
    mga = MicroGeneticAlgorithm(
        fitness_functions,
        constraint_functions,
        num_variables=2,
        population_size=50,
        agents_to_keep=10,
        agents_to_shuffle=5,
        random_restarts=10,
        max_iterations=500,
        num_bits=128,
        random_seed=42,
    )
    mga.run_iterations()


if __name__ == "__main__":
    main()
