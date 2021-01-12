# Nature-inspired-metaheuristics : Implement Nature-Inspired Metaheuristics to optimize NP-Hard 0/1 Knapsack Problem

## Introduction 
Nature Inspired Computing is a branch of Computer Science that strives to develop and improve computing techniques by observing how nature behaves to solve complex problems.  Nature Inspired Computing has led to groundbreaking research and created new branches like neural networks, evolutionary computation, and artificial intelligence. 

Swarm Optimization is a computational method that optimizes a problem by generating a population of candidate solutions and moving these candidates in a search space according to a mathematical formula.  Eventually the swarm moves to the best solutions. Bat Algorithm is based on swarm intelligence heuristic search algorithm.  It is a metaheuristic algorithm for global optimization. The algorithm is inspired by the echolocation of bats when hunting for prey, with varying loudness, frequency, pulse rates of emission, distance, and speed. The bat algorithm can be implemented on combinatorial optimization problems. These are problems where an optimal solution must be identified from a finite set of solutions. These solutions are either discrete or can be made discrete.

0-1 Knapsack Problem is problem of NP-Hard class. It is also a combinatorial optimization problem. Problems in NP-Hard (Non-Deterministic Polynomial-Time Hardness) classes are those which are at least as hard as the hardest problems in NP. It is not a decision problem, which can be answered with a ‘Yes’ or ‘No’. In 0-1 knapsack problem, given a set of items, each with a different weight and price, the goal is to choose items to get the maximum possible price, while at the same not exceed the maximum weight limit of the knapsack. In 0-1 knapsack the binary numbers 0 and 1 are the decision variables that decide which item gets selected to be put into the bag.1 means the item is chosen, while 0 means the item is not chosen. Hence, the general Bat Algorithm will have to be slightly modified to discretize the random solutions that will be generated.

Nature Inspired Computing is an important field that efficiently helps implement nature inspired algorithms onto optimization problems. In this report, we will be implementing Binary Bat Algorithm to 0-1 Knapsack Problem to find the optimal solution.

## Bat Algorithm
The Bat Algorithm is a nature inspired algorithm which was developed by Xin-She Yang in 2010. It is based on the hunting behavior of bats at night. Bats use echolocation to detect prey, avoid obstacles and locate their resting location. These bats emit high-pitched sounds and interpret their echoes to determine the distance and direction of targets. Each bat has a frequency (f), position (x), velocity (v), loudness (A) and pulse rate (r).


## 0/1 Knapsack Problem
0/1 Knapsack Problem or Binary Knapsack Problem is a NP-Hard combinatorial problem that aims to maximize profits from a knapsack without exceeding its maximum capacity. Items of different price and weight will have to be selected to be put in the knapsack while maintaining a weight that doesn’t exceed a constant and at the same time obtain the maximum possible total price.

## Experimental Setup
The experimental results are collected using the following setup. Two types of data were used for the experiment. A Large-Scale Data Consisting of 21 testcases of large-scale data, each containing thousands of data points with weight and price attributes and the integer optimal price of each of the 21 testcases. The second data used is Low Dimensional Data consisting of 10 testcases of large-scale data, each containing thousands of data points with weight and price attributes. It also contains the integer optimal price of each of the 10 testcases. The program was coded using Python language and libraries like NumPy, pandas, OS and matplotlib using the software Visual Studio Code. We then clean the data in a way that helps the program make use of it efficiently.  

## Conclusion
Nature provides solutions to many of the problems that we have today. We can take inspiration from nature for developing problem solving techniques. The Bat Algorithm is one of many nature inspired algorithms (Ant Colony, Grey Wolf, Genetic) that help solve optimization problems, specifically combinatorial optimization problems. 
In this report, we have successfully implemented Bat Algorithm to find the optimal solution to 0/1 knapsack problem with a 99.71% accuracy.

