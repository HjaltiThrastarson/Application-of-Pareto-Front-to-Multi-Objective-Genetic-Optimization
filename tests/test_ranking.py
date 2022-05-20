from cgi import test
from omfe.ranking import NonDominatedSort


def test_dominance_ranking_one_objective():
    # Setup
    f_1 = lambda x: x**2
    ranking_algo = NonDominatedSort([f_1])
    agents = [(-30), (-2), (1), (2), (5)]

    # Execute
    ranking = ranking_algo.non_dominated_sort(agents)

    # Compare
    expected_ranking = [{1}, {2, -2}, {5}, {-30}]
    assert ranking == expected_ranking


def test_dominance_ranking_two_objectives():
    # Setup
    f_1 = lambda agent: agent[0] ** 2 + agent[1] ** 2
    f_2 = lambda agent: agent[0] + agent[1]
    ranking_algo = NonDominatedSort([f_1, f_2])
    agents = [(-1, -3), (1, 3), (-1, 3), (3, -1)]

    # Execute
    ranking = ranking_algo.non_dominated_sort(agents)

    # Compare
    expected_ranking = [{(-1, -3)}, {(-1, 3), (3, -1)}, {(1, 3)}]
    assert ranking == expected_ranking
