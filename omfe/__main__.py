from .problems import ChankongHaimes
from .mga import MicroGeneticAlgorithm


def main():
    problem = ChankongHaimes()
    mga = MicroGeneticAlgorithm(
        problem,
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
