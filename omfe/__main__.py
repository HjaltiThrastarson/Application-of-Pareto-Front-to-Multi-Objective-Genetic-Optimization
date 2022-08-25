import os
from datetime import datetime
import numpy as np
import omfe
import matplotlib as mpl
import matplotlib.pyplot as plt
from omfe.evaluator import Evaluator, NonDominatedSortEvaluator, WeightBasedEvaluator

from omfe.mga import AlgorithmRunner, MicroGeneticAlgorithm
from omfe.problems import (
    CTP1,
    BinhKorn,
    ChankongHaimes,
    Kursawe,
    Problem,
    plot_agents,
    plot_agent,
)


def main():
    plot_sum()
    plot_squares()
    plot_pareto_dominance_ex()
    compare_plots(BinhKorn())
    compare_plots(ChankongHaimes())
    compare_plots(CTP1())
    compare_plots(Kursawe())

    show_moo_iterations(BinhKorn())
    show_moo_iterations(ChankongHaimes())
    show_moo_iterations(CTP1())
    show_moo_iterations(Kursawe())

    show_soo_iterations(BinhKorn())
    show_soo_iterations(ChankongHaimes())
    show_soo_iterations(CTP1())
    show_soo_iterations(Kursawe())


def plot_sum():
    fix, axes = plt.subplots()
    axes.set_ylabel("a*c + (1-a)*s")
    axes.set_ylim(0, 3)
    axes.set_xlim(-3, 3)
    x = np.linspace(-5, 5, 1000)
    y1 = (x - 1) ** 2
    y2 = (x + 1) ** 2
    # y3 = 0.6 * (x - 1) ** 2 + 0.4 * (x + 1) ** 2
    y4 = 0.8 * (x - 1) ** 2 + 0.2 * (x + 1) ** 2
    # y5 = 0.4 * (x - 1) ** 2 + 0.6 * (x + 1) ** 2
    y6 = 0.2 * (x - 1) ** 2 + 0.8 * (x + 1) ** 2
    y7 = 0.5 * (x - 1) ** 2 + 0.5 * (x + 1) ** 2

    axes.plot(x, y1)
    axes.plot(x, y2)
    # axes.plot(x, y3)
    axes.plot(x, y4)
    # axes.plot(x, y5)
    axes.plot(x, y6)
    axes.plot(x, y7)

    axes.scatter(-1, 0, color="red")
    axes.scatter(1, 0, color="red")
    axes.scatter(0, 1, color="red")

    # axes.scatter(x[np.argmin(y3)], min(y3), color="red")
    axes.scatter(x[np.argmin(y4)], min(y4), color="red")
    # axes.scatter(x[np.argmin(y5)], min(y5), color="red")
    axes.scatter(x[np.argmin(y6)], min(y6), color="red")

    plt.savefig("out/x_squares_min.png")
    plt.cla()


def plot_squares():
    fig, axes = plt.subplots()
    axes.set_xlabel("Angle")
    axes.set_ylabel("Cost")
    axes.set_xlim(-1.2, 1.2)
    axes.set_ylim(0, 1.2)
    x = np.linspace(-1, 1, 100)
    y = x**2
    axes.plot(x, y)
    axes.scatter(0, 0, color="red")
    plt.savefig("out/x_square.png")
    plt.cla()


def plot_pareto_dominance_ex():
    # Point
    fig, axes = plt.subplots()
    axes.set_xlabel("Size")
    axes.set_ylabel("Cost")
    axes.set_title(f"Objective Space Plot")
    axes.set_xlim(0, 1.2)
    axes.set_ylim(0, 1.2)
    f1 = np.array([0.6])
    f2 = np.array([0.6])
    axes.scatter(f1, f2)
    axes.vlines(0.6, 0, 0.6, linestyle="dashed")
    axes.hlines(0.6, 0, 0.6, linestyle="dashed")
    plt.savefig("out/pareto-1.png")
    plt.cla()

    # Worse
    axes.set_xlabel("Size")
    axes.set_ylabel("Cost")
    axes.set_title(f"Objective Space Plot")
    axes.set_xlim(0, 1.2)
    axes.set_ylim(0, 1.2)
    f1 = np.array([0.6])
    f2 = np.array([0.6])
    f1_2 = np.array([1])
    f2_2 = np.array([1])
    axes.scatter(f1, f2)
    axes.scatter(f1_2, f2_2, color="red", marker="^")
    axes.vlines(0.6, 0, 0.6, linestyle="dashed")
    axes.hlines(0.6, 0, 0.6, linestyle="dashed")
    axes.vlines(f1_2, 0, f2_2, linestyle="dashed", color="red")
    axes.hlines(f2_2, 0, f1_2, linestyle="dashed", color="red")
    plt.savefig("out/pareto-2.png")
    plt.cla()

    # better
    axes.set_xlabel("Size")
    axes.set_ylabel("Cost")
    axes.set_title(f"Objective Space Plot")
    axes.set_xlim(0, 1.2)
    axes.set_ylim(0, 1.2)
    f1 = np.array([0.6, 1])
    f2 = np.array([0.6, 1])
    f1_2 = np.array([0.6])
    f2_2 = np.array([0.4])
    axes.scatter(f1, f2)
    axes.scatter(f1_2, f2_2, color="red", marker="^")
    axes.vlines(0.6, 0, 0.6, linestyle="dashed")
    axes.hlines(0.6, 0, 0.6, linestyle="dashed")
    axes.vlines(f1_2, 0, f2_2, linestyle="dashed", color="red")
    axes.hlines(f2_2, 0, f1_2, linestyle="dashed", color="red")
    plt.savefig("out/pareto-3.png")
    plt.cla()

    # still better
    axes.set_xlabel("Size")
    axes.set_ylabel("Cost")
    axes.set_title(f"Objective Space Plot")
    axes.set_xlim(0, 1.2)
    axes.set_ylim(0, 1.2)
    f1 = np.array([0.6, 1, 0.6])
    f2 = np.array([0.6, 1, 0.4])
    f1_2 = np.array([0.4])
    f2_2 = np.array([0.4])
    axes.scatter(f1, f2)
    axes.scatter(f1_2, f2_2, color="red", marker="^")
    axes.vlines(0.6, 0, 0.6, linestyle="dashed")
    axes.hlines(0.6, 0, 0.6, linestyle="dashed")
    axes.vlines(f1_2, 0, f2_2, linestyle="dashed", color="red")
    axes.hlines(f2_2, 0, f1_2, linestyle="dashed", color="red")
    plt.savefig("out/pareto-4.png")
    plt.cla()

    # Not dominated
    axes.set_xlabel("Size")
    axes.set_ylabel("Cost")
    axes.set_title(f"Objective Space Plot")
    axes.set_xlim(0, 1.2)
    axes.set_ylim(0, 1.2)
    f1 = np.array([0.6, 1, 0.4, 0.6])
    f2 = np.array([0.6, 1, 0.4, 0.4])
    f1_2 = np.array([0.8])
    f2_2 = np.array([0.2])
    axes.scatter(f1, f2)
    axes.scatter(f1_2, f2_2, color="red", marker="^")
    axes.vlines(0.6, 0, 0.6, linestyle="dashed")
    axes.hlines(0.6, 0, 0.6, linestyle="dashed")
    axes.vlines(f1_2, 0, f2_2, linestyle="dashed", color="red")
    axes.hlines(f2_2, 0, f1_2, linestyle="dashed", color="red")
    plt.savefig("out/pareto-5.png")
    plt.cla()

    # Bigger example
    axes.set_xlabel("Size")
    axes.set_ylabel("Cost")
    axes.set_title(f"Objective Space Plot")
    axes.set_xlim(0, 1.2)
    axes.set_ylim(0, 1.2)
    f1 = np.array([1, 0.2, 0.6, 1, 0.7, 0.4, 0.6])
    f2 = np.array([0.4, 0.8, 0.6, 1, 0.8, 1, 0.4])
    f1_2 = np.array([0.2, 0.4, 0.8])
    f2_2 = np.array([0.6, 0.4, 0.2])
    axes.scatter(f1, f2)
    axes.scatter(f1_2, f2_2, color="red", marker="^")
    # axes.vlines(f1, 0, f2, linestyle="dashed")
    # axes.hlines(f2, 0, f1, linestyle="dashed")
    # axes.vlines(f1_2, 0, f2_2, linestyle="dashed", color="red")
    # axes.hlines(f2_2, 0, f1_2, linestyle="dashed", color="red")
    plt.savefig("out/pareto-6.png")
    plt.cla()


def show_soo_iterations(problem: Problem, seed=42):
    weight_evaluator = WeightBasedEvaluator(problem, seed)
    mga_weight = MicroGeneticAlgorithm(
        problem,
        weight_evaluator,
        population_size=5,
        agents_to_keep=1,
        max_iterations=15,
        num_bits=8,
        seed=seed,
    )
    runner = AlgorithmRunner(mga_weight, seed)
    times = 5
    all_agents = runner.run_all(times)

    time = datetime.today().strftime("%y%m%d-%H%M%S")
    outdir = f"out/{time}-{problem.__class__.__name__}-SOO"
    os.mkdir(outdir)

    for i_random_restart in range(0, times):
        for j_iteration in range(0, mga_weight.max_iterations, 1):
            fig, ax = plt.subplots()
            plot_agents(ax, all_agents[i_random_restart][j_iteration], problem)
            problem.plot_pareto_front(ax)
            for k_random_restart in range(0, i_random_restart):
                plot_agent(ax, all_agents[k_random_restart][-1][-1], problem)
            ax.set_title(
                f"{problem}, Population: {mga_weight.population_size}, Random Restart: {i_random_restart}, Iteration: {j_iteration}"
            )
            plt.savefig(f"{outdir}/{i_random_restart}-{j_iteration}.png")
            plt.close()


def show_moo_iterations(problem: Problem, seed=42):
    """Plot every iteration of the problem"""
    pareto_evaluator = NonDominatedSortEvaluator(problem)
    mga_pareto = MicroGeneticAlgorithm(
        problem,
        pareto_evaluator,
        population_size=201,
        agents_to_keep=1,
        max_iterations=200,
        num_bits=8,
        seed=seed,
    )
    population_history = mga_pareto.run()

    time = datetime.today().strftime("%y%m%d-%H%M%S")
    outdir = f"out/{time}-{problem.__class__.__name__}-MOO"
    os.mkdir(outdir)
    for i in range(0, len(population_history), 10):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 600)
        ax.set_ylim(-350, 200)
        plot_agents(ax, population_history[i], problem)
        problem.plot_pareto_front(ax)
        ax.set_title(
            f"{problem}, Population: {mga_pareto.population_size}, Iteration: {i}"
        )
        plt.savefig(f"{outdir}/{i}.png")
        plt.close()


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
    ax[0].set_title(f"SOO. Population: {mga_weight.population_size}, Restarts: {times}")

    plot_agents(ax[1], final_population, problem)
    problem.plot_pareto_front(ax[1])
    ax[1].set_title(f"MOO. Population: {mga_pareto.population_size}")
    fig.suptitle(f"{problem}")
    time = datetime.today().strftime("%y%m%d-%H%M%S")
    outname = f"out/{time}-{problem.__class__.__name__}-MOO"
    plt.savefig(f"{outname}.png")
    plt.close()


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
