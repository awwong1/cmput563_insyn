#!/usr/bin/env python3

import numpy as np
import copy
import utilities as utils


"""
matlab stores data by column order, numpy stores by row order

q   matlab: (4, 8, 2)   python: (2, 4, 8)       Structure of HHMM
PI  matlab: (8, 8, 3)   python: (3, 8, 8)       Initial state distribution (Vertical)
A   matlab: (8, 8, 3)   python: (3, 8, 8)       Transition probabilities (Horizontal)
B   matlab: (4, 7, 8)   python: (8, 4, 7)       Observation probabilities (Emissions)
"""
maxIter = 100
maxError = 1.0e-03
depth = 4
width = 8
np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)

alphabet = np.array([i for i in range(1, 9)])

# The values inside the matrix correspond to
q = np.zeros((2, depth, width))
# 0 -> No state present at the level
# 1 -> State
# 2 -> Signifies terminal state which recurses back to parent root
q[0] = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                 [1, 1, 2, 0, 0, 0, 0, 0],
                 [1, 1, 2, 1, 1, 2, 0, 0],
                 [1, 1, 1, 2, 1, 1, 1, 2]])
# 0 -> No children
# x -> x children (in order of occurrence from the above) are assigned to the node
q[1] = np.array([[3, 0, 0, 0, 0, 0, 0, 0],
                 [3, 3, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 4, 4, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]])

testSeq = np.array([[1, 3, 1, 3, 2, 5, 8, 4, 2, 5, 4, 6, 7, 2, 8],
                    [7, 3, 5, 8, 3, 5, 2, 3, 7, 5, 4, 6, 3, 2, 5]])

# Here, we need to subtract 1 so we can index from alphabet literal to index
testSeq = testSeq - 1
alphabet = alphabet - 1

allY, allX = np.where(q[0, :, :] == 1)
# Indices of production states, which are nodes which do not have children
prodY, prodX = np.where((q[0, :, :] == 1) & (q[1, :, :] == 0))
# Indices of internal states (states which have children)
intY, intX = np.where(q[1, :, :] != 0)

np.random.seed(0)
IS_RANDOM = False


# Vertical Transitions
def init_vertial_transitions():
    PI = np.zeros((depth - 1, width, width))
    for i in range(len(allX)):
        # Skip the first order (main root)
        if allY[i] != 0:
            # Walk backwards summing nodes that match that the parent says how many children it has
            parent_i, = np.where(np.cumsum(q[1, allY[i] - 1, :]) >= allX[i] + 1)
            parent_i = parent_i[0]
            PI[allY[i] - 1, parent_i, allX[i]] = 1 + (np.random.random_sample() - 0.5) / 5
            if not IS_RANDOM:
                PI[allY[i] - 1, parent_i, allX[i]] = 1 + (0.65 - 0.5) / 5

    return PI



# Horizontal Transitions
def init_horizontal_transitions(PI):
    A = np.zeros((depth - 1, width, width))
    for i in range(len(allX)):
        if allY[i] != 0:
            parent_i,  = np.where(np.cumsum(q[1, allY[i] - 1, :]) >= allX[i] + 1)
            parent_i = parent_i[0]
            jArray, = np.where(PI[allY[i] - 1, parent_i, :] != 0)
            jArray = np.append(jArray, jArray[-1] + 1)
            A[allY[i] - 1, allX[i], jArray] = 1 + (np.random.random_sample(jArray.shape[0]) - 0.5) / 5
            if not IS_RANDOM:
                A[allY[i] - 1, allX[i], jArray] = 1 + (np.array([0.45 for i in range(jArray.shape[0])]) - 0.5) / 5

    return A


# Emissions
def init_emissions():
    B = np.zeros((width, depth, width - 1))
    for i in range(len(prodX)):
        r = np.ones((len(alphabet))) + (np.random.random_sample(len(alphabet)) - 0.5) / 5
        if not IS_RANDOM:
            r = np.ones((len(alphabet))) + (np.array([0.60 for i in range(len(alphabet))]) - 0.5) / 5
        B[:, prodY[i], prodX[i]] = r / np.sum(r)

    return B


def train():
    PI = init_vertial_transitions()
    A = init_horizontal_transitions(PI)
    B = init_emissions()

    # Standarize
    for row in range(width):
        for col in range(depth - 1):
            A[col, row, :] = np.nan_to_num(A[col, row, :] / np.sum(A[col, row, :]))
            PI[col, row, :] = np.nan_to_num(PI[col, row, :] / np.sum(PI[col, row, :]))

    # Matlab deepcopies by default
    initA = copy.deepcopy(A)
    initB = copy.deepcopy(B)
    initPI = copy.deepcopy(PI)

    Palt = 0
    stop = 0

    for iteration in range(maxIter):
        print('Training iteration {}'.format(iteration))
        ergA = np.zeros(initA.shape)
        ergB = np.zeros(initB.shape)
        ergPI = np.zeros(initPI.shape)

        ergAVis = np.zeros(initA.shape)
        ergBVis = np.zeros(initB.shape)
        ergPIVis = np.zeros(initPI.shape)

        Pact = 0

        # Over each row of input
        for s in range(testSeq.shape[0]):
            seq = testSeq[s, :]
            # why 0 here, its a never used arg ...
            PI, A, B, P = utils.HHMM_EM(q, seq, initA, initPI, initB, alphabet, 0)

            Pact = Pact + P
            B = np.nan_to_num(B)
            A = np.nan_to_num(A)
            PI = np.nan_to_num(PI)

            ergA = ergA + A
            ergB = ergB + B
            ergPI = ergPI + PI


        # Standardize?
        for i in range(len(prodX)):
            ergBVis[:, prodY[i], prodX[i]] = np.nan_to_num(ergB[:, prodY[i], prodX[i]] / np.sum(ergB[:, prodY[i], prodX[i]]))

        # Standardize?
        for row in range(width):
            for col in range(depth - 1):
                ergAVis[col, row, :] = np.nan_to_num(ergA[col, row, :] / np.sum(ergA[col, row, :]))
                ergPIVis[col, row, :] = np.nan_to_num(ergPI[col, row, :] / np.sum(ergPI[col, row, :]))

        # Code to draw output.....

        # This may not be exactly the same as matlab
        if abs(Pact - Palt) / (1 + abs(Palt)) < maxError:
            # Mask out q[0] where 1 if 1, 0 else
            states = np.where(q[0] == 1, 1, 0)
            if np.linalg.norm(ergAVis.flatten() - initA.flatten(), np.inf) / np.sum(states.flatten()) < maxError:
                print('Convergence after {} iterations'.format(iteration))
                stop = 1

        if stop == 0:
            Palt = Pact
            initA = ergAVis
            initB = ergBVis
            initPI = ergPIVis
        else:
            break

    # Finished
    return ergPIVis, ergAVis, ergBVis


if __name__ == '__main__':
    PI, A, B = train()

    print(A)


