import numpy as np
import copy

def logSum(a, b):
    if np.isnan(a):
        return b
    elif np.isnan(b):
        return a

    if np.isneginf(a) and np.isneginf(b):
        return np.NINF

    return max(a, b) + np.log(1 + np.exp(-abs(a - b)))


def logProd(a, b):
    if np.isnan(a):
        return a
    elif np.isnan(b):
        return b

    if np.isneginf(a) and np.isneginf(b):
        return np.NINF

    return a + b


def findSArray(q, d, i):
    if i == 0:
        # start = 1
        start = 0
        end = np.cumsum(q[1, d, 0:(i + 1)]).astype(int)
        end = end[-1]
    else:
        start = np.cumsum(q[1, d, 0:i]).astype(int)
        end = np.cumsum(q[1, d, 0:(i + 1)]).astype(int)
        # start = start[-1] + 1
        start = start[-1]
        end = end[-1]

    return np.array([j for j in range(start, end)])


def findJArray(q, d, i):
    jArr, = np.where(q[0, d, :] == 2)
    preJ, = np.where(jArr < i)
    # TODO: Not safe code, but I think we can asssume jArray is non-empty
    postJ, = np.where(jArr >= i)
    postJ = postJ[0]

    # If preJ is empty
    if preJ.size == 0:
        preJ = np.append(preJ, 0)
        # start = preJ[-1] + 1
        start = preJ[-1]
    else:
        # start = jArr[preJ[-1]] + 1
        start = jArr[preJ[-1]] + 1

    end = jArr[postJ]

    return np.array([i for i in range(start, end + 1)])


def expectationEtaLog(A, PI, q, alpha, beta, seq):
    """
    q       matlab: (4, 8, 2)           python: (2, 4, 8)
    PI      matlab: (8, 8, 3)           python: (3, 8, 8)
    A       matlab: (8, 8, 3)           python: (3, 8, 8)
    B       matlab: (4, 7, 8)           python: (8, 4, 7)
    alpha   matlab: (15, 15, 4, 8)      python: (4, 8, 15, 15)
    beta    matlab: (15, 15, 4, 8)      python: (4, 8, 15, 15)
    eta_in  matlab: (15, 4, 8)          python: (8, 15, 4)
    eta_out matlab: (15, 4, 8)          python: (8, 15, 4)
    xi      matlab: (15, 4, 7, 4, 8)    python: (7, 4, 8, 15, 4)
    seq (1,15)
    """

    allY, allX = np.where(q[0, :, :] == 1)
    eta_in = np.full((q.shape[2], len(seq), q.shape[1]), np.NINF)
    eta_out = np.full((q.shape[2], len(seq), q.shape[1]), np.NINF)

    # eta_in
    sArray = findSArray(q, 0, 0)
    for i in sArray[:-1]:
        for t in range(len(seq)):
            if t == 0:
                eta_in[i, t, 1] = PI[0, 0, i]
            else:
                log_sum = np.nan
                jArray = findJArray(q, 1, i)
                for j in jArray:
                    log_pro = logProd(alpha[1, j, 0, t - 1], A[0, j, i])
                    log_sum = logSum(log_sum, log_pro)

                eta_in[i, t, 1] = log_sum

    for d in range(2, max(allY) + 1):
        for i in range(np.max(allX) + 1):
            if q[0, d, i] == 1:
                parent, = np.where(np.cumsum(q[1, d - 1, :]) >= i + 1)
                parent = parent[0]
                jArray = findJArray(q, d, i)

                for t in range(len(seq)):
                    if t == 0:
                        eta_in[i, t, d] = logProd(eta_in[parent, t, d-1], PI[d-1, parent, i])
                    else:
                        outer_sum = np.nan
                        for ttick in range(t):
                            inner_sum = np.nan
                            for j in jArray:
                                log_pro = logProd(alpha[d, j, ttick, t-1], A[d - 1, j, i])
                                inner_sum = logSum(inner_sum, log_pro)

                            outer_sum = logSum(outer_sum, logProd(inner_sum, eta_in[parent, ttick, d-1]))
                        eta_in[i, t, d] = logSum(outer_sum, logProd(eta_in[parent, t, d - 1], PI[d-1, parent, i]))

    # eta_out
    sArray = findSArray(q, 0, 0)
    for i in sArray[:-1]:
        for t in range(len(seq) - 1, -1, -1):
            if t == len(seq) - 1:
                end = sArray[-1]
                eta_out[i, t, 1] = A[0, i, end]
            else:
                log_sum = np.nan
                jArray = findJArray(q, 1, i)
                for j in jArray:
                    log_pro = logProd(beta[1, j, t + 1, len(seq) - 1], A[0, i, j])
                    log_sum = logSum(log_sum, log_pro)

                eta_out[i, t, 1] = log_sum

    for d in range(2, max(allY) + 1):
        for i in range(np.max(allX) + 1):
            if q[0, d, i] == 1:
                parent, = np.where(np.cumsum(q[1, d - 1, :]) >= i + 1)
                parent = parent[0]
                jArray = findJArray(q, d, i)
                e, = np.where(q[0, d, i:] == 2)
                e = e[0] + i

                for t in range(len(seq) - 1, -1, -1):
                    if t == len(seq) - 1:
                        eta_out[i, t, d] = logProd(eta_out[parent, t, d-1], A[d-1, i, e])
                    else:
                        outer_sum = np.nan
                        for k in range(t+1, len(seq)):
                            inner_sum = np.nan
                            for j in jArray:
                                log_pro = logProd(beta[d, j, t + 1, k], A[d - 1, i, j])
                                inner_sum = logSum(inner_sum, log_pro)

                            outer_sum = logSum(outer_sum, logProd(inner_sum, eta_out[parent, k, d-1]))
                        eta_out[i, t, d] = logSum(outer_sum, logProd(eta_out[parent, t, d - 1], A[d-1, i, e]))


    return eta_in, eta_out


def expectationXiChiLog(A, PI, q, eta_in, eta_out, alpha, beta, seq):
    """
    q       matlab: (4, 8, 2)           python: (2, 4, 8)
    PI      matlab: (8, 8, 3)           python: (3, 8, 8)
    A       matlab: (8, 8, 3)           python: (3, 8, 8)
    B       matlab: (4, 7, 8)           python: (8, 4, 7)
    alpha   matlab: (15, 15, 4, 8)      python: (4, 8, 15, 15)
    beta    matlab: (15, 15, 4, 8)      python: (4, 8, 15, 15)
    eta_in  matlab: (15, 4, 8)          python: (8, 15, 4)
    eta_out matlab: (15, 4, 8)          python: (8, 15, 4)
    xi      matlab: (15, 4, 7, 4, 8)    python: (7, 4, 8, 15, 4)
    seq (1,15)
    """

    # Top right corner of (2, :, 1, 15)
    temp = np.array([alpha[1, i, 0, len(seq) - 1] for i in range(alpha.shape[1])])
    POlambda = np.log(np.sum(np.exp(temp)))

    allY, allX = np.where(q[0, :, :] == 1)

    xi = np.full((q.shape[2] - 1, q.shape[1], q.shape[2], len(seq), q.shape[1]), np.NINF)
    chi = np.full((q.shape[2] - 1, len(seq), q.shape[1]), np.NINF)

    gamma_in = copy.deepcopy(chi)
    gamma_out = copy.deepcopy(chi)

    # xi
    for t in range(len(seq)):
        sArray = findSArray(q, 0, 0)
        for i in sArray[:-1]:
            jArray = findJArray(q, 1, i)
            # All but the last element
            for k in jArray[:-1]:
                if t == len(seq) - 1:
                    temp = logProd(alpha[1, i, 0, t], A[0, i, jArray[-1]])
                    xi[i, 1, jArray[-1], t, 1] = logProd(temp, -POlambda)
                else:
                    temp = logProd(alpha[1, i, 0, t], beta[1, k, t+1, len(seq) - 1])
                    temp = logProd(temp, A[0, i, k])
                    xi[i, 1, k, t, 1] = logProd(temp, -POlambda)


    for t in range(len(seq)):
        for d in range(2, max(allY) + 1):
            for i in range(np.max(allX) + 1):
                if q[0, d, i] == 1:
                    parent, = np.where(np.cumsum(q[1, d - 1, :]) >= i + 1)
                    parent = parent[0]
                    jArray = findJArray(q, d, i)
                    for j in jArray:
                        if j == jArray[-1]:
                            log_sum = np.nan
                            for s in range(t+1):
                                log_pro = logProd(eta_in[parent, s, d - 1], alpha[d, i, s, t])
                                log_sum = logSum(log_sum, log_pro)

                            temp = logProd(log_sum, A[d - 1, i, j])
                            temp = logProd(temp, eta_out[parent, t, d - 1])
                            xi[i, d, j, t, d] = logProd(temp, -POlambda)
                        else:
                            if t != len(seq) - 1:
                                log_sum1 = np.nan
                                for s in range(0, t+1):
                                    log_pro1 = logProd(eta_in[parent, s, d - 1], alpha[d, i, s, t])
                                    log_sum1 = logSum(log_sum1, log_pro1)

                                log_sum2 = np.nan
                                for e in range(t+1, len(seq)):
                                    log_pro2 = logProd(eta_out[parent, e, d - 1], beta[d, j, t + 1, e])
                                    log_sum2 = logSum(log_sum2, log_pro2)

                                temp = logProd(log_sum1, log_sum2)
                                temp = logProd(temp, A[d - 1, i, j])
                                xi[i, d, j, t, d] = logProd(temp, -POlambda)

    # chi
    sArray = findSArray(q, 0, 0)
    for i in sArray[:-1]:
        temp = logProd(PI[0, 0, i], beta[1, i, 0, len(seq) - 1])
        chi[i, 0, 1] = logProd(temp, -POlambda)

    for t in range(len(seq)):
        for d in range(2, max(allY) + 1):
            for i in range(np.max(allX) + 1):
                if q[0, d, i] == 1:
                    parent, = np.where(np.cumsum(q[1, d - 1, :]) >= i + 1)
                    parent = parent[0]

                    log_sum = np.nan
                    for m in range(len(seq)):
                        log_pro = logProd(beta[d, i, t, m], eta_out[parent, m, d - 1])
                        log_sum = logSum(log_sum, log_pro)

                    temp = logProd(log_sum, eta_in[parent, t, d - 1])
                    temp = logProd(temp, PI[d - 1, parent, i])
                    chi[i, t, d] = logProd(temp, -POlambda)


    # gamma_in
    for d in range(1, max(allY) + 1):
        for i in range(np.max(allX) + 1):
            if q[0, d, i] == 1:
                jArray = findJArray(q, d, i)
                for t in range(1, len(seq)):
                    log_sum = np.nan
                    for k in jArray[:-1]:
                        log_sum = logSum(log_sum, xi[k, d, i, t-1, d])
                    gamma_in[i, t, d] = log_sum

    # gamma_out
    for d in range(1, max(allY) + 1):
        for i in range(np.max(allX) + 1):
            if q[0, d, i] == 1:
                jArray = findJArray(q, d, i)
                for t in range(len(seq)):
                    log_sum = np.nan
                    for k in jArray:
                        log_sum = logSum(log_sum, xi[i, d, k, t, d])
                    gamma_out[i, t, d] = log_sum


    return xi, chi, gamma_in, gamma_out


def expectationAlphaBetaLog(A, PI, q, B, seq):
    """
    Forward-Backward algorithm:
        - alpha: probability of ending up in a particcular state given first t observations,
                 denoted forward pass P(X_t | o_{1:t})
        - beta:  probability of observing the remaining observations given any starting point,
                 denoted backward pass P(o_{t+1:T} | X_t)
        - Combine to get P(X_t | o_{1:T}) ~ P(o_{t+1:T} | X_t) P(X_t | o_{1:t})

    q       matlab: (4, 8, 2)           python: (2, 4, 8)
    PI      matlab: (8, 8, 3)           python: (3, 8, 8)
    A       matlab: (8, 8, 3)           python: (3, 8, 8)
    B       matlab: (4, 7, 8)           python: (8, 4, 7)
    alpha   matlab: (15, 15, 4, 8)      python: (4, 8, 15, 15)
    beta    matlab: (15, 15, 4, 8)      python: (4, 8, 15, 15)
    eta_in  matlab: (15, 4, 8)          python: (8, 15, 4)
    eta_out matlab: (15, 4, 8)          python: (8, 15, 4)
    xi      matlab: (15, 4, 7, 4, 8)    python: (7, 4, 8, 15, 4)
    seq (1,15)
    """

    # Indices of production states
    prodY, prodX = np.where((q[0, :, :] == 1) & (q[1, :, :] == 0))

    # Indices of internal states
    intY, intX = np.where(q[1, :, :] != 0)

    # Initialize alpha and beta
    alpha = np.full((q.shape[1], q.shape[2], len(seq), len(seq)), np.NINF)
    beta = np.full((q.shape[1], q.shape[2], len(seq), len(seq)), np.NINF)

    # Forward pass -> alpha
    # Here we step backwards
    for t in range(len(seq) - 1, -1, -1):
        for k in range(-1, len(seq) - t - 1):
            for d in range(np.max(prodY), 0, -1):
                for i in range(np.max(prodX) + 1):
                    if q[1, d, i] == 0 and q[0, d, i] == 1:
                        parent, = np.where(np.cumsum(q[1, d - 1, :]) >= i + 1)
                        parent = parent[0]
                        jArray = findJArray(q, d, i)

                        if k == -1:
                            alpha[d, i, t, t] = logProd(PI[d - 1, parent, i], B[seq[t], d, i])
                        else:
                            log_sum = np.nan
                            # Production states under parent node
                            for j in jArray[:-1]:
                                log_pro = logProd(alpha[d, j, t, t + k], A[d - 1, j, i])
                                log_sum = logSum(log_sum, log_pro)

                            alpha[d, i, t, t + k + 1] = logProd(log_sum, B[seq[t + k + 1], d, i])

            for d in range(np.max(intY), 0, -1):
                for i in range(np.max(intX) + 1):
                    if q[1, d, i] != 0:
                        parent, = np.where(np.cumsum(q[1, d - 1, :]) >= i + 1)
                        parent = parent[0]
                        jArray = findJArray(q, d, i)
                        sArray = findSArray(q, d, i)

                        if k == -1:
                            log_sum = np.nan
                            for s in sArray[:-1]:
                                log_pro = logProd(alpha[d + 1, s, t, t], A[d, s, sArray[-1]])
                                log_sum = logSum(log_sum, log_pro)
                            alpha[d, i, t, t] = logProd(log_sum, PI[d - 1, parent, i])
                        else:
                            outer_sum = np.nan
                            for ell in range(k+1):
                                log_sum1 = np.nan
                                for j in jArray[:-1]:
                                    log_pro1 = logProd(alpha[d, j, t, t + ell], A[d - 1, j, i])
                                    log_sum1 = logSum(log_sum1, log_pro1)

                                log_sum2 = np.nan
                                for s in sArray[:-1]:
                                    log_pro2 = logProd(alpha[d + 1, s, t + ell + 1, t + k + 1], A[d, s, sArray[-1]])
                                    log_sum2 = logSum(log_sum2, log_pro2)

                                outer_sum = logSum(outer_sum, logProd(log_sum1, log_sum2))

                            log_sum3 = np.nan
                            for s in sArray[:-1]:
                                log_pro3 = logProd(alpha[d + 1, s, t, t + k + 1], A[d, s, sArray[-1]])
                                log_sum3 = logSum(log_sum3, log_pro3)

                            alpha[d, i, t, t + k + 1] = logSum(outer_sum, logProd(log_sum3, PI[d - 1, parent, i]))


            # Beta
            for d in range(np.max(prodY), 0, -1):
                for i in range(np.max(prodX) + 1):
                    if q[1, d, i] == 0 and q[0, d, i] == 1:
                        jArray = findJArray(q, d, i)
                        e, = np.where(q[0, d, i:] == 2)
                        e = e[0] + i
                        if k == -1:
                            beta[d, i, t, t] = logProd(B[seq[t], d, i], A[d-1, i, e])
                        else:
                            log_sum = np.nan
                            # Production states under parent node
                            for j in jArray[:-1]:
                                log_pro = logProd(beta[d, j, t + 1, t + k + 1], A[d - 1, i, j])
                                log_sum = logSum(log_sum, log_pro)

                            beta[d, i, t, t + k + 1] = logProd(log_sum, B[seq[t], d, i])

            for d in range(np.max(intY), 0, -1):
                for i in range(np.max(intX) + 1):
                    if q[1, d, i] != 0:
                        jArray = findJArray(q, d, i)
                        sArray = findSArray(q, d, i)

                        if k == -1:
                            log_sum = np.nan
                            for s in sArray[:-1]:
                                log_pro = logProd(beta[d + 1, s, t, t], PI[d, i, s])
                                log_sum = logSum(log_sum, log_pro)
                            beta[d, i, t, t] = logProd(log_sum, A[d - 1, i, jArray[-1]])
                        else:
                            outer_sum = np.nan
                            for ell in range(k+1):
                                log_sum1 = np.nan
                                for s in sArray[:-1]:
                                    log_pro1 = logProd(beta[d + 1, s, t, t + ell], PI[d, i, s])
                                    log_sum1 = logSum(log_sum1, log_pro1)

                                log_sum2 = np.nan
                                for j in jArray[:-1]:
                                    log_pro2 = logProd(beta[d, j, t + ell + 1, t + k + 1], A[d - 1, i, j])
                                    log_sum2 = logSum(log_sum2, log_pro2)

                                outer_sum = logSum(outer_sum, logProd(log_sum1, log_sum2))

                            log_sum3 = np.nan
                            for s in sArray[:-1]:
                                log_pro3 = logProd(beta[d + 1, s, t, t + k + 1], PI[d, i, s])
                                log_sum3 = logSum(log_sum3, log_pro3)

                            beta[d, i, t, t + k + 1] = logSum(outer_sum, logProd(log_sum3, A[d - 1, i, jArray[-1]]))

    return alpha, beta


def estimationLog(q, xi, chi, gamma_in, gamma_out, seq, B, alpha):
    """
    q       matlab: (4, 8, 2)           python: (2, 4, 8)
    PI      matlab: (8, 8, 3)           python: (3, 8, 8)
    A       matlab: (8, 8, 3)           python: (3, 8, 8)
    B       matlab: (4, 7, 8)           python: (8, 4, 7)
    alpha   matlab: (15, 15, 4, 8)      python: (4, 8, 15, 15)
    beta    matlab: (15, 15, 4, 8)      python: (4, 8, 15, 15)
    eta_in  matlab: (15, 4, 8)          python: (8, 15, 4)
    eta_out matlab: (15, 4, 8)          python: (8, 15, 4)
    xi      matlab: (15, 4, 7, 4, 8)    python: (7, 4, 8, 15, 4)
    seq (1,15)
    """

    # Indices of production states
    prodY, prodX = np.where((q[0, :, :] == 1) & (q[1, :, :] == 0))
    allY, allX = np.where(q[0, :, :] == 1)

    PI_new = np.full((q.shape[1] - 1, q.shape[2], q.shape[2]), np.NINF)
    A_new = np.full((q.shape[1] - 1, q.shape[2], q.shape[2]), np.NINF)
    B_new = np.full((q.shape[2], q.shape[1], q.shape[2] - 1), np.NINF)

    # vertical transitions
    sArray = findSArray(q, 0, 0)
    for i in sArray[:-1]:
        log_sum = np.nan
        for k in sArray[:-1]:
            log_sum = logSum(log_sum, chi[k, 0, 1])
        PI_new[0, 0, i] = logProd(chi[i, 0, 1], -log_sum)

    for d in range(2, max(allY) + 1):
        for i in range(np.max(allX) + 1):
            if q[0, d, i] == 1:
                parent, = np.where(np.cumsum(q[1, d - 1, :]) >= i + 1)
                parent = parent[0]
                jArray = findJArray(q, d, i)
                sum_above = np.nan

                for t in range(len(seq)):
                    sum_above = logSum(sum_above, chi[i, t, d])

                sum_below = np.nan
                for m in jArray[:-1]:
                    log_sum = np.nan
                    for t in range(len(seq)):
                        log_sum = logSum(log_sum, chi[m, t, d])
                    sum_below = logSum(sum_below, log_sum)

                PI_new[d - 1, parent, i] = logProd(sum_above, -sum_below)


    # horizontal transitions
    for d in range(1, max(allY) + 1):
        for i in range(np.max(allX) + 1):
            if q[0, d, i] == 1:
                jArray = findJArray(q, d, i)
                for j in jArray:
                    sum_above = np.nan
                    for t in range(len(seq)):
                        sum_above = logSum(sum_above, xi[i, d, j, t, d])

                    sum_below = np.nan
                    for t in range(len(seq)):
                        sum_below = logSum(sum_below, gamma_out[i, t, d])

                    A_new[d - 1, i, j] = logProd(sum_above, -sum_below)
                    if np.isinf(A_new[d - 1, i, j]):
                        A_new[d - 1, i, j] = np.log(np.finfo(float).eps)


    # emissions
    for v in alpha:
        for d in range(1, max(prodY) + 1):
            for i in range(max(prodX) + 1):
                if q[1, d, i] == 0 and q[0, d, i] == 1:
                    d1 = np.nan
                    for n in range(len(seq)):
                        if seq[n] == v:
                            d1 = logSum(d1, chi[i, n, d])

                    d2 = np.nan
                    for n in range(1, len(seq)):
                        if seq[n] == v:
                            d2 = logSum(d2, gamma_in[i, n, d])

                    d3 = np.nan
                    for n in range(len(seq)):
                        d3 = logSum(d3, chi[i, n, d])

                    d4 = np.nan
                    for n in range(1, len(seq)):
                        d4 = logSum(d4, gamma_in[i, n, d])

                    x1 = logSum(d1, d2)
                    x2 = logSum(d3, d4)
                    B_new[v, d, i] = logProd(x1, -x2)

    # standardization
    A_new = np.exp(A_new)
    PI_new = np.exp(PI_new)
    B_new = np.exp(B_new)




    return PI_new, A_new, B_new


def HHMM_EM(q, seq, A, PI, B, alph, Palt):
    # Expectation
    np.seterr(divide='ignore')
    A = np.log(A)
    B = np.log(B)
    PI = np.log(PI)
    np.seterr(divide='warn')

    alpha, beta = expectationAlphaBetaLog(A, PI, q, B, seq)
    temp = np.array([alpha[1, i, 0, len(seq) - 1] for i in range(alpha.shape[1])])
    P = np.sum(np.exp(temp))

    eta_in, eta_out = expectationEtaLog(A, PI, q, alpha, beta, seq)
    xi, chi, gamma_in, gamma_out = expectationXiChiLog(A, PI, q, eta_in, eta_out, alpha, beta, seq)

    # Maximization
    PI_new, A_new, B_new = estimationLog(q, xi, chi, gamma_in, gamma_out, seq, B, alph)

    return PI_new, A_new, B_new, P
