# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=consider-using-enumerate


import numpy as np
from mga import MicroGeneticAlgorithm
from problems import ChankongHaimes, Problem
from evaluator import NonDominatedSortEvaluator
from evaluator import WeightBasedEvaluator
from evaluator import Evaluator


class TournamentSelector:
    def __init__(
        self,
        mga: MicroGeneticAlgorithm,
        evaluator: Evaluator,
        tournament_size=10,
        num_tournaments=50,
        num_iterations=100,
    ):
        # User defined variables
        self.mga = mga
        self.evaluator = evaluator
        self.tournament_size = tournament_size
        self.num_tournaments = num_tournaments
        self.num_iterations = num_iterations

        # Internal variables
        print("RUNNING SETUP")
        self.mga.run_iterations(print_progress=False)
        self.contenders = self.mga.best_agents
        self.winners = np.empty((0, self.mga.problem.num_variables))
        # Generate index book for random choice
        self.contenders_index = self.generate_indexes()
        print("SETUP FINISHED")

    def generate_indexes(self):
        contenders_index = np.empty(self.contenders.shape[0], dtype=int)
        for i in range(len(contenders_index)):
            contenders_index[i] = i
        return contenders_index

    def generate_tournament(self):
        # Generate a random tournament
        tournament_indexes = np.random.choice(
            self.contenders_index, self.tournament_size
        )
        tournament = np.empty((self.tournament_size, self.mga.problem.num_variables))
        for entry in range(self.tournament_size):
            tournament[entry] = self.contenders[tournament_indexes[entry]]
        return tournament

    def run_tournament(self, tournament):
        # Run the tournament
        tournament_agents, tournament_fitness = self.evaluator.evaluate_agents(
            tournament
        )
        # This line kind of assumes that the evaluator is a NonDominatedSortEvaluator
        winners = tournament_agents[tournament_fitness == 0]
        # THIS IS A VERY SLOW LINE, BUT IT IS HARD TO ESTIMATE NUMBERS OF WINNERS
        self.winners = np.append(self.winners, winners, axis=0)

    def run_iterations(self, print_progress=True):
        # Run the tournaments
        for i in range(self.num_iterations):
            for j in range(self.num_tournaments):
                tournament = self.generate_tournament()
                self.run_tournament(tournament)
            self.contenders = self.winners
            self.winners = np.empty((0, self.mga.problem.num_variables))
            self.contenders_index = self.generate_indexes()
            if print_progress:
                print(f"Iteration {i+1}, number of winners: {len(self.contenders)}")


def main():
    problem = ChankongHaimes()
    evaluator = WeightBasedEvaluator(problem)
    tournament_evaluator = NonDominatedSortEvaluator(problem)
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
    TMS = TournamentSelector(
        MGA,
        tournament_evaluator,
        tournament_size=20,
        num_tournaments=100,
        num_iterations=20,
    )
    TMS.run_iterations()


if __name__ == "__main__":
    main()
