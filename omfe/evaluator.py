# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=consider-using-enumerate


import numpy as np

from problems import Problem
from util_functions import get_random_vec_with_sum_one


class Evaluator:
    def __init__(self, problem: Problem):
        self.problem = problem

    def evaluate_agents(self, agents):
        """
        Function to call to return a list of agents sorted by their fitness/ranking
        returns:
            agents_sorted: list of agents sorted by their fitness/ranking
            fitness or None if not applicable
        """
        return agents, None

    def reset(self):
        """
        Function to call to reset the evaluator
        """
        pass

    def info(self):
        """
        Function to return a string with information about the evaluator
        """
        return "Nothing was done"

    def compare_agents(self, best_fitness, best_agent, fitness, agent):
        """
        Function to compare which is best, might be different for different evaluators
        """
        return best_fitness > fitness


class WeightBasedEvaluator(Evaluator):
    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)
        self.weights = get_random_vec_with_sum_one(len(self.problem.functions))

    def calculate_fitness_functions(self, agents):
        """
        Calculates fitness of all agents based on the weighted approach
        """
        fitness_all = np.zeros(len(agents))
        for i in range(len(agents)):
            # Calculate fitness for current agent and apply weights
            fitness_list = np.array(self.problem.evaluate_functions(agents[i]))
            fitness = np.sum(self.weights * fitness_list)
            # Calculate if the constraints are broken and if so, set fitness to inf
            if np.all(self.problem.evaluate_constraints(agents[i])):
                pass
            else:
                fitness = np.inf
            fitness_all[i] = fitness
        return fitness_all

    def evaluate_agents(self, agents):
        """
        Evaluates the agents according to some fitness/ranking function and returns a sorted
        list based on the ranking function.
        """
        fitness = self.calculate_fitness_functions(agents)
        return agents[np.argsort(fitness)], fitness[np.argsort(fitness)]

    def reset(self) -> None:
        self.weights = get_random_vec_with_sum_one(len(self.problem.functions))

    def info(self) -> str:
        return f"Weights: {self.weights}"


class NonDominatedSortEvaluator(Evaluator):
    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)
        self.objectives = problem.functions

    def breaks_constraints(self, agent):
        if np.all(self.problem.evaluate_constraints(agent)) == False:
            return True
        return False

    def is_dominated(self, agent_a, agent_b):
        """
        Return true if agent_a is dominated by agent_b
        """

        a_fitness = np.array(self.problem.evaluate_functions(agent_a))
        b_fitness = np.array(self.problem.evaluate_functions(agent_b))
        b_not_worse_than_a = np.all(a_fitness >= b_fitness)
        b_better_than_a_in_one_objective = np.any(a_fitness > b_fitness)

        return b_not_worse_than_a and b_better_than_a_in_one_objective

    def evaluate_agents(self, agents):
        """
        Sorts the agents based on the non-dominance of the objective functions
        """
        # Initialize the domination counts aka fitness
        fitness = np.zeros(len(agents))
        # Loop through all agents
        for i in range(len(agents)):
            if self.breaks_constraints(agents[i]):
                fitness[i] = np.inf
                continue
            # Loop through all other agents
            for j in range(i + 1, len(agents)):
                # Check if the agent is dominated by the other agent
                if self.is_dominated(agents[i], agents[j]):
                    # If so, increment the domination count of the other agent
                    fitness[i] += 1
                # Check if the other agent is dominated by the agent
                elif self.is_dominated(agents[j], agents[i]):
                    # If so, increment the domination count of the agent
                    fitness[j] += 1

        return agents[np.argsort(fitness)], fitness[np.argsort(fitness)]

    def compare_agents(self, best_fitness, best_agent, fitness, agent):
        """
        Checks if the best agent is dominated by the new one
        """
        return self.is_dominated(best_agent, agent) or self.breaks_constraints(
            best_agent
        )

    def info(self):
        return "Non-dominated sorted"
