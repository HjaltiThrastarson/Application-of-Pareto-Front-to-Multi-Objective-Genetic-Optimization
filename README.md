# Application-of-Pareto-Front-to-Multi-Objective-Genetic-Optimization

## TO-DO

1. Figure out how to implement shuffling with grey code with the Mga [**Hjalti**]
2. ~~Set up a class skeleton with a random start and calculate fitness functionality~~
3. ~~reate fitness function~~
4. Create an evaluate function that runs one iteration and shuffles with the mga[**Hjalti**]
5. Create a vizualuzation for fitness output i.e [**Max**]
6. Setup a library to have as a benchmark [**Max**]
    * Evaluate existing libraries
    * Setup MWE
7. **LONG TERM** Look into how to create a pareto front fitness function [**Lukas + Max + Hjalti**]
8. Look into other representations for shuffling other than gray code [**Lukas**]
9. Find out how to set weights, like normalization ? [**Lukas**]

## IDEAS

1. Implement micro genetic algorithm
2. Use Chankong and Haimes test func for evaluation
3. The range is [-20, 20], idea is to map small ranges in that range to binary numbers, for example with 1 bit we would have 0 as -20 and 1 as 20

![mga](mga.png)

## Literature
* [Multi-Objective Optimization (Wiki)](https://en.wikipedia.org/wiki/Multi-objective_optimization)
* [Pareto-Efficiency (Wiki)](https://en.wikipedia.org/wiki/Pareto_efficiency)
* [A tutorial on multiobjective optimization: fundamentals and evolutionary methods](https://link.springer.com/article/10.1007/s11047-018-9685-y)
* [Multi-objective Optimisation Using Evolutionary Algorithms: An Introduction](https://link.springer.com/chapter/10.1007/978-0-85729-652-8_1)
* [Adaptive weighted sum method for multiobjective optimization: a new method for Pareto front generation](https://link.springer.com/article/10.1007/s00158-005-0557-6)
* [The weighted sum method for multi-objective optimization: new insights](https://link.springer.com/article/10.1007/s00158-009-0460-7)

## Frameworks/Libraries/Resources
* [scikit-opt](https://github.com/guofei9987/scikit-opt)
* [pymoo](https://github.com/anyoptimization/pymoo) - [Corresponding Paper](https://ieeexplore.ieee.org/abstract/document/9078759)
* [Test functions for optimizations (Wikipedia)](https://en.wikipedia.org/wiki/Test_functions_for_optimization)