# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=consider-using-enumerate


from logging import exception
import numpy as np
from numpy.typing import NDArray
import sys
from problems import ChankongHaimes
from graymapping import reversegray_mapping
from graymapping import gray_mapping


class MicroGeneticAlgorithm:
    def __init__(
        self,
        problem,
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

        self.problem = problem
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
        self.weights = get_random_vec_with_sum_one(length=len(self.problem.functions))

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

    def initialize_agent(self, only_valid = False):
        """
        Initializes a single agent that doesn't break constraints if set to only_valid
        """
        constraints_broken = True
        agent = np.zeros((self.num_variables))
        while constraints_broken:
            for variables in range(self.num_variables):
                agent[variables] = np.random.uniform(
                    self.problem.search_domain[variables][0],
                    self.problem.search_domain[variables][1],
                )
            if only_valid is not True:
                break
            if np.all(self.problem.evaluate_constraints(agent)):
                constraints_broken = False
        return agent

    def initialize_agents(self):
        ## Random restart for agents
        agents = np.zeros((self.population_size, self.num_variables))
        for i in range(self.population_size):
            agents[i] = self.initialize_agent()
        return agents

    def calculate_fitness_function(self, random_restart, iteration):
        """
        Calculate fitness of all agents
        """
        fitness = 0
        for i in range(len(self.agents)):
            fitness_list = np.array(self.problem.evaluate_functions(self.agents[i]))
            fitness = np.sum(self.weights * fitness_list)
            # Calculate fitness for each constraint and add them up
            if np.all(self.problem.evaluate_constraints(self.agents[i])):
                pass
            else:
                fitness += 1000000
            self.fitness[i] = fitness
            if fitness < self.best_fitness[random_restart]:
                self.best_fitness[random_restart] = fitness
                self.best_agents[random_restart][:] = self.agents[i]
            self.fitness_history[random_restart][iteration][i] = fitness
            self.agents_history[random_restart][iteration][i] = self.agents[i]

    def evaluate_agents(self):
        """
        Sorts agents based on fitness and keeps the best ones
        """
        # This is a magic line that sorts the agents by fitness
        #agents_sorted = np.array([x for _, x in sorted(zip(self.fitness, self.agents))])
        agents_sorted = self.agents[self.fitness.argsort()]
        return agents_sorted[: self.agents_to_keep]

    def shuffle_agents(self, agents_to_keep):
        ## Shuffle agents, generate a random cutoff point and
        ## select the best self.number_to_shuffle agents to be trailing agents
        new_agents = np.zeros((self.population_size, self.num_variables))
        agents_to_keep_fully = self.agents_to_keep - self.agents_to_shuffle
        new_agents[:agents_to_keep_fully] = agents_to_keep[:agents_to_keep_fully]
        # Generate random cutoff point
        # Set first agent to keep fully to trailing agent
        trailing_agents = 1
        for i in range(agents_to_keep_fully, self.agents_to_keep):
            for j in range(self.num_variables):
                cutoff = np.random.randint(0, self.num_bits)
                # Don't know how to do without for loop, might be a better way
                for h in range(agents_to_keep_fully):
                    if trailing_agents == h:
                        if h == trailing_agents:
                            trailing_agents += 1
                            # Map trailing and leading to graycode
                            gray_code_mapping_leading = gray_mapping(
                                agents_to_keep[i][j],
                                self.num_bits,
                                self.problem.search_domain[j]
                            )
                            grat_code_mapping_trailing = gray_mapping(
                                agents_to_keep[trailing_agents - 1][j],
                                self.num_bits,
                                self.problem.search_domain[j]
                            )
                            # Combine them to make a new agent
                            new_agent = (
                                gray_code_mapping_leading[:cutoff]
                                + grat_code_mapping_trailing[cutoff:]
                            )
                            new_agent = reversegray_mapping(int(new_agent, 2),
                                                            self.num_bits,
                                                            self.problem.search_domain[j]
                                                            )
                            new_agents[i][j] = new_agent
                            break
                        if trailing_agents == agents_to_keep_fully:
                            trailing_agents = 1

        # Randomly generate the rest of the agents
        for i in range(self.agents_to_keep, self.population_size):
            new_agents[i] = self.initialize_agent()
        self.agents = new_agents

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
            self.weights = get_random_vec_with_sum_one(
                length=len(self.problem.functions)
            )


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
    mga = MicroGeneticAlgorithm(
        problem,
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
