import numpy as np
import random
from selection_functions import roulette_wheel_selection
from mutation import mutate
from crossover import crossover


def MinOne(x):
    global NFE
    NFE += 1
    n_var = len(x)
    indices1 = np.arange(0, n_var, 2)
    indices2 = np.arange(1, n_var, 2)
    z = sum(x[indices1])-sum(x[indices2])
    return z


def main():
    global NFE
    NFE = 0

    CostFunction = MinOne  # Cost Function

    n_var = 20  # Number of Decision Variables

    VarSize = np.array([1, n_var])  # Decision Variables Matrix Size

    # GA Parameters

    MaxIt = 500  # Maximum Number of Iterations

    nPop = 10  # Population Size

    pc = 0.8  # crossover Percentage
    nc = 2 * round(pc * nPop / 2)  # Number of Offsprings (Parnets)

    pm = 0.1  # Mutation Percentage
    nm = round(pm * nPop)  # Number of Mutants

    mu = 0.1  # Mutation Rate

    beta = 8


    class EmptyIndividual():
        pass

    Best = [EmptyIndividual() for i in range(MaxIt)]
    pop = np.array([EmptyIndividual() for i in range(nPop)])

    for i in range(nPop):
        # Initialize Position
        pop[i].Position = np.array([random.randint(0, 1) for i in range(n_var)])

        # Evaluation
        pop[i].Cost = CostFunction(pop[i].Position)

    # Sort Population
    Costs = np.array([pop[i].Cost for i in range(nPop)])
    SortOrder = np.argsort(Costs)

    pop = pop[SortOrder]

    # Array to Hold Best Cost Values
    BestCost = np.zeros([MaxIt, 1])

    # Store Cost
    WorstCost = pop[-1].Cost

    # Array to Hold Number of Function Evaluations
    nfe = np.zeros([MaxIt, 1])

    for it in range(MaxIt):

        # Calculate Selection Probabilities
        P = np.expm1(-beta * Costs / WorstCost)
        P = P / sum(P)

        # crossover
        popc = np.array([[EmptyIndividual() for i in range(int(nc / 2))]
                         , [EmptyIndividual() for i in range(int(nc / 2))]]).transpose()

        for k in range(int(nc / 2)):

            i1 = roulette_wheel_selection(P)
            i2 = roulette_wheel_selection(P)

            # Select Parents
            p1 = pop[i1]
            p2 = pop[i2]

            # Apply crossover
            popc[k, 0].Position, popc[k, 1].Position = crossover(p1.Position, p2.Position)

            # Evaluate Offsprings
            popc[k, 0].Cost = CostFunction(popc[k, 0].Position)
            popc[k, 1].Cost = CostFunction(popc[k, 1].Position)

        popc = np.array([popc[i] for i in range(len(popc))]).transpose()
        popc = np.concatenate((popc[0], popc[1]), axis=0)

        # Mutation
        popm = np.array([EmptyIndividual() for i in range(nm)])

        for k in range(nm):
            # Select Parent
            i = random.randint(0, nPop - 1)
            p = pop[i]

            # Apply Mutation
            popm[k].Position = mutate(np.array(p.Position), mu)

            # Evaluate Mutant
            popm[k].Cost = CostFunction(popm[k].Position)

        # Create Merged Population
        pop = np.concatenate((pop, popc, popm), axis=0)

        # Sort Population
        Costs = np.array([pop[i].Cost for i in range(len(pop))])
        SortOrder = np.argsort(Costs)
        Costs = Costs[SortOrder]
        pop = pop[SortOrder]

        # Truncation
        pop = pop[0:nPop]
        Costs = Costs[0:nPop]

        # Store Best Cost Ever Found
        BestCost[it] = pop[0].Cost
        Best[it] = pop[0]

        # Store Worst Cost Ever Found
        WorstCost = pop[-1].Cost

        # Store NFE
        nfe[it] = NFE

        # Show Iteration Information
        print('Iteration', it, ': NFE = ', nfe[it], ', Best Cost = ', BestCost[it], 'Chromosome: ', pop[0].Position)


if __name__ == '__main__':
    main()
