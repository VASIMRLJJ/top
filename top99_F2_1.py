import numpy as np
import matplotlib.pyplot as plt


def top(nelx, nely, volfrac, penal, rmin):
    x = np.full((nely, nelx), volfrac)
    loop = 0
    change = 1.0

    while change > 0.01:
        loop += 1
        xold = x

        U = FE(nelx, nely, x, penal)
        KE = lk()
        c = 0.0
        dc = np.zeros((nely, nelx))
        for elx in range(1, nelx+1):
            for ely in range(1, nely+1):
                n1 = (nely + 1) * (elx - 1) + ely
                n2 = (nely + 1) * elx + ely
                dc[ely - 1, elx - 1] = 0
                for i in range(2):
                    edof = np.array([2 * n1 - 1, 2 * n1, 2 * n2 - 1, 2 * n2,
                            2 * n2 + 1, 2 * n2 + 2, 2 * n1 + 1, 2 * n1 + 2]).T
                    Ue = U[edof-1, i]
                    c = c + (x[ely-1, elx-1] ** penal)*Ue.conj().T.dot(KE.dot(Ue))
                    dc[ely-1, elx-1] = dc[ely-1, elx-1] - (penal*x[ely-1, elx-1] ** (penal-1))*Ue.conj().T.dot(KE.dot(Ue))
        dc = check(nelx, nely, rmin, x, dc)
        x = OC(nelx, nely, x, volfrac, dc)
        change = abs(x - xold).max()
        print(change)
        plt.matshow(x)
        plt.show()


def OC(nelx, nely, x, volfrac, dc):
    l1 = 0
    l2 = 100000
    move = 0.2
    xnew = 0
    while l2 - l1 > 1e-4:
        lmid = (l2+l1)/2
        xnew = np.maximum(0.001, np.maximum(x - move, np.minimum(1., np.minimum(x + move, x * np.sqrt(-dc / lmid)))))
        if np.sum(np.sum(xnew)) - volfrac * nelx * nely > 0:
            l1 = lmid
        else:
            l2 = lmid
    return xnew


def check(nelx, nely, rmin, x, dc):
    dcn = np.zeros((nely, nelx))
    for i in range(1, nelx+1):
        for j in range(1, nely+1):
            sum = 0.0
            for k in range(int(max(i-np.floor(rmin), 1)),
                           int(min(i+np.floor(rmin), nelx))+1):
                for l in range(int(max(j - np.floor(rmin), 1)),
                               int(min(j + np.floor(rmin), nely)) + 1):
                    fac = rmin - np.sqrt((i - k) ** 2 + (j - l) ** 2)
                    sum = sum + max(0, fac)
                    dcn[j-1, i-1] = dcn[j-1, i-1] + max(0, fac) * x[l-1, k-1] * dc[l-1, k-1]
            dcn[j-1, i-1] /= x[j-1, i-1] * sum
    return dcn


def FE(nelx, nely, x, penal):
    KE = lk()
    K = np.zeros((2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1)))
    F = np.zeros((2*(nely+1)*(nelx+1), 2))
    U = np.zeros((2 * (nely + 1) * (nelx + 1), 2))
    for elx in range(1, nelx+1):
        for ely in range(1, nely+1):
            n1 = (nely + 1) * (elx - 1) + ely
            n2 = (nely + 1) * elx + ely
            edof = np.array([2 * n1 - 1, 2 * n1, 2 * n2 - 1, 2 * n2,
                    2 * n2 + 1, 2 * n2 + 2, 2 * n1 + 1, 2 * n1 + 2]).T
            K[np.ix_(edof-1, edof-1)] = K[np.ix_(edof-1, edof-1)] + x[ely-1, elx-1] ** penal * KE

    F[2*(nelx+1)*(nely+1)-1, 0] = -1.0
    F[2*nelx*(nely+1)+1, 1] = 1.0

    fixeddofs = np.array([2*(nelx+1)*(nely+1)-1])
    alldofs = np.array(range(2 * (nely + 1) * (nelx + 1)))
    freedofs = np.setdiff1d(alldofs, fixeddofs)

    U[np.ix_(freedofs)] = np.linalg.solve(K[np.ix_(freedofs, freedofs)], F[np.ix_(freedofs)])
    U[np.ix_(fixeddofs)] = np.zeros(U[np.ix_(fixeddofs)].shape)
    return U


def lk():
    E = 1.0
    nu = 0.3
    k = [1 / 2 - nu / 6, 1 / 8 + nu / 8, - 1 / 4 - nu / 12, - 1 / 8 + 3 * nu / 8,
         - 1 / 4 + nu / 12, - 1 / 8 - nu / 8, nu / 6, 1 / 8 - 3 * nu / 8]
    KE = [[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
          [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
          [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
          [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
          [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
          [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
          [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
          [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]]
    KE = np.array(KE)
    KE *= E/nu**2
    return KE


if __name__ == '__main__':
    top(30, 30, 0.4, 3.0, 1.2)
