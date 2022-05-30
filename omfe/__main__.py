from omfe.problems import ChankongHaimes, BinhKorn
from omfe.mga import AlgorithmRunner, MicroGeneticAlgorithm
from omfe.ranking import WeightedSumSort
from omfe.visualize import plot_agents

import matplotlib as mpl
import matplotlib.pyplot as plt
import time


def main():
    seed = 2
    # problem = ChankongHaimes()
    problem = BinhKorn()
    ranking = WeightedSumSort(objectives=problem.functions, seed=seed, weights=None)
    mga = MicroGeneticAlgorithm(
        problem=problem,
        sorter=ranking,
        population_size=5,
        agents_to_keep=1,
        max_iterations=200,
        num_bits=8,
        seed=seed,
    )
    agents_history = mga.run()

    fig, ax = plt.subplots()  # Convenience method to create figure and plot
    for agents in agents_history:
        plot_agents(
            axes=ax,
            agents=agents,
            problem=mga.problem,
            param_dict={"marker": "o"},
        )
        fig.show()
    # runner = AlgorithmRunner(mga, random_seed=seed)
    # best_agents = runner.run(100)
    # print(best_agents)


if __name__ == "__main__":
    main()
