from problems import ChankongHaimes
from mga import MicroGeneticAlgorithm
from evaluator import WeightBasedEvaluator
from Visualize import FitPlot, FitHistRec, VarPlot, VarHistRec


def main():
    problem = ChankongHaimes()
    evaluator = WeightBasedEvaluator(problem)
    mga = MicroGeneticAlgorithm(
        problem,
        evaluator=evaluator,
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
