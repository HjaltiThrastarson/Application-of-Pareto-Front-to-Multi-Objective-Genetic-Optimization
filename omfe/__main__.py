import omfe
import matplotlib as mpl
import matplotlib.pyplot as plt
from omfe.evaluator import Evaluator, NonDominatedSortEvaluator, WeightBasedEvaluator

from omfe.mga import AlgorithmRunner, MicroGeneticAlgorithm
from omfe.problems import CTP1, BinhKorn, ChankongHaimes, Kursawe, Problem, plot_agents


def main():
    compare_plots(BinhKorn())
    compare_plots(ChankongHaimes())
    compare_plots(CTP1())
    compare_plots(Kursawe())
    plt.show()


def compare_plots(problem: Problem, seed=42):
    # Reduce MOO problem to SOO problem by using weights for different
    # objective functions to obtain single solution.
    # Run multiple times with random weights to explore MOO objective space
    weight_evaluator = WeightBasedEvaluator(problem, seed)
    mga_weight = MicroGeneticAlgorithm(
        problem,
        weight_evaluator,
        population_size=5,
        agents_to_keep=1,
        max_iterations=50,
        num_bits=8,
        seed=seed,
    )
    runner = AlgorithmRunner(mga_weight, seed)
    times = 200
    best_agents = runner.run(times)

    # Now run MOO problem with Pareto front based ranking to find multiple
    # solutions at the same time
    pareto_evaluator = NonDominatedSortEvaluator(problem)
    mga_pareto = MicroGeneticAlgorithm(
        problem,
        pareto_evaluator,
        population_size=201,
        agents_to_keep=1,
        max_iterations=50,
        num_bits=8,
        seed=seed,
    )
    final_population = mga_pareto.run()[-1]

    # Plot both next to each other
    fig, ax = plt.subplots(1, 2)
    plot_agents(ax[0], best_agents, problem)
    problem.plot_pareto_front(ax[0])
    ax[0].set_title(
        f"{problem} SOO with random weights. Population Size: {mga_weight.population_size}, Random Restarts: {times}"
    )

    plot_agents(ax[1], final_population, problem)
    problem.plot_pareto_front(ax[1])
    ax[1].set_title(
        f"{problem} MOO with pareto front ranking. Population Size: {mga_pareto.population_size}"
    )


def tournament():
    problem = omfe.problems.ChankongHaimes()
    evaluator = omfe.evaluator.WeightBasedEvaluator(problem)
    tournament_evaluator = omfe.evaluator.NonDominatedSortEvaluator(problem)
    MGA = omfe.mga.MicroGeneticAlgorithm(
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
    TMS = omfe.tournament_selector.TournamentSelector(
        MGA,
        tournament_evaluator,
        tournament_size=20,
        num_tournaments=100,
        num_iterations=20,
    )
    TMS.run_iterations()


if __name__ == "__main__":
    main()
