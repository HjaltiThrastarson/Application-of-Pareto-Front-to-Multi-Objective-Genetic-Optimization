# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=consider-using-enumerate


import numpy as np
from numpy.typing import NDArray
from problems import ChankongHaimes, Problem
from evaluator import WeightBasedEvaluator
from evaluator import NonDominatedSortEvaluator
from evaluator import Evaluator
from util_functions import reversegray_mapping
from util_functions import gray_mapping


class MicroGeneticAlgorithm:
    def __init__(
        self,
        problem: Problem,
        evaluator: Evaluator,
        population_size=50,
        agents_to_keep=10,
        agents_to_shuffle=8,
        random_restarts=10,
        max_iterations=100,
        iteration_tolerance=10,
        num_bits=8,
        random_seed=42,
    ):

        ## User input class variables

        self.problem = problem
        self.evaluator = evaluator
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
        # How many max iterations per random restart to do
        self.max_iterations = max_iterations
        # How many iterations without improvement to tolerate before restarting
        self.iteration_tolerance = iteration_tolerance
        # Number of bits to use in gray_mapping, more --> slower but more accurate
        self.num_bits = num_bits
        # Set random seed for generating agents
        np.random.seed(random_seed)

        ## Internal class variables

        # Current Fitness of all agents and corresponding variables
        self.fitness = np.zeros(population_size)
        # Fitness history, so that we can plot it
        self.fitness_history = np.zeros(
            (random_restarts, max_iterations, population_size)
        )
        # Initial random start for agents
        self.agents = self.initialize_agents()
        # Agent history
        self.agents_history = np.zeros(
            (
                random_restarts,
                max_iterations,
                population_size,
                self.problem.num_variables,
            )
        )
        self.best_fitness = np.ones((random_restarts)) * np.inf
        self.best_agents = np.zeros((random_restarts, self.problem.num_variables))

    def initialize_agent(self, only_valid=False):
        """
        Initializes a single agent that doesn't break constraints if set to only_valid
        """
        constraints_broken = True
        agent = np.zeros((self.problem.num_variables))
        while constraints_broken:
            for variables in range(self.problem.num_variables):
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
        agents = np.zeros((self.population_size, self.problem.num_variables))
        for i in range(self.population_size):
            agents[i] = self.initialize_agent()
        return agents

    def shuffle_agents(self, agents_to_keep):
        ## Shuffle agents, generate a random cutoff point and
        ## select the best self.number_to_shuffle agents to be trailing agents
        new_agents = np.zeros((self.population_size, self.problem.num_variables))
        agents_to_keep_fully = self.agents_to_keep - self.agents_to_shuffle
        new_agents[:agents_to_keep_fully] = agents_to_keep[:agents_to_keep_fully]
        # Generate random cutoff point
        # Set first agent to keep fully to trailing agent
        trailing_agents = 1
        for i in range(agents_to_keep_fully, self.agents_to_keep):
            for j in range(self.problem.num_variables):
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
                                self.problem.search_domain[j],
                            )
                            grat_code_mapping_trailing = gray_mapping(
                                agents_to_keep[trailing_agents - 1][j],
                                self.num_bits,
                                self.problem.search_domain[j],
                            )
                            # Combine them to make a new agent
                            new_agent = (
                                gray_code_mapping_leading[:cutoff]
                                + grat_code_mapping_trailing[cutoff:]
                            )
                            new_agent = reversegray_mapping(
                                int(new_agent, 2),
                                self.num_bits,
                                self.problem.search_domain[j],
                            )
                            new_agents[i][j] = new_agent
                            break
                        if trailing_agents == agents_to_keep_fully:
                            trailing_agents = 1

        # Randomly generate the rest of the agents
        for i in range(self.agents_to_keep, self.population_size):
            new_agents[i] = self.initialize_agent()
        self.agents = new_agents

    def run_iterations(self, print_progress=True):
        """
        Function that runs the iterations and random restarts of the algorithm

        This is the only function that should be run by the user
        """

        for random_restart in range(self.random_restarts):
            improve_count = 0
            for iteration in range(self.max_iterations):
                # Call evaluator to get agents/fitness sorted
                agents_sorted, fitness = self.evaluator.evaluate_agents(self.agents)
                # Set history of agents and fitness, sorted by performance
                self.agents_history[random_restart, iteration] = agents_sorted
                if fitness is not None:
                    self.fitness_history[random_restart, iteration] = fitness
                    # Set best fitness/agent
                    if self.evaluator.compare_agents(
                        self.best_fitness[random_restart],
                        self.best_agents[random_restart],
                        fitness[0],
                        agents_sorted[0],
                    ):
                        self.best_fitness[random_restart] = fitness[0]
                        self.best_agents[random_restart] = agents_sorted[0]
                    else:
                        improve_count += 1
                    if improve_count >= self.iteration_tolerance:
                        break
                # Shuffle the agents based on the graymapping mga approach
                self.shuffle_agents(agents_sorted[: self.agents_to_keep])
            if print_progress:
                print(
                    f"Random restart/iterations {random_restart} {iteration+1} done \
                    best fitness: {self.best_fitness[random_restart]} \
                    {self.evaluator.info()}"
                )
            self.evaluator.reset()


def main():
    problem = ChankongHaimes()
    evaluator = WeightBasedEvaluator(problem)
    MGA = MicroGeneticAlgorithm(
        problem,
        evaluator,
        population_size=10,
        agents_to_keep=5,
        agents_to_shuffle=4,
        random_restarts=1000,
        max_iterations=1000,
        iteration_tolerance=10,
        num_bits=64,
        random_seed=0,
    )
    MGA.run_iterations()


if __name__ == "__main__":
    main()
