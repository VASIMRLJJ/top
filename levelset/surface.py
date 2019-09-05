import numpy as np
from scipy import sparse as ss
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from CSRBF import CSRBF


class Surface:
    def __init__(self, nelx: int, nely: int):
        self.nelx = nelx
        self.nely = nely
        self.csrbf = CSRBF(3.5, 0, 0)
        self.coordinate = self.coordinate()
        # self.A = self.matrixA()

    def coordinate(self):
        c = []
        for nx in range(self.nelx+1):
            for ny in range(self.nely+1):
                c.append((nx, ny))
        return np.array(c)

    def matrixA(self):
        A = np.array([])
        loop = 0
        for (X, Y) in self.coordinate:
            self.csrbf.xi, self.csrbf.yi = (X, Y)
            c = self.csrbf.c2(range(self.nelx+1), range(self.nely+1))
            loop += 1
            if not loop % 10:
                print('matrixA init:', int(loop/10), '/', int((self.nelx+1)*(self.nely+1)/10))
            a = c.T.flatten()
            a = np.array([a]).T
            if len(A):
                A = np.c_[A, a]
            else:
                A = a
        return ss.dia_matrix(A)

    def origin(self):
        x = range(self.nelx+1)
        y = range(self.nely+1)
        x, y = np.meshgrid(x, y)
        z1 = np.minimum(np.minimum(x, self.nelx+1-x), np.minimum(y, self.nely+1-y)).T
        x = np.repeat(np.array([x]), 17, axis=0).transpose((2, 1, 0))
        y = np.repeat(np.array([y]), 17, axis=0).transpose((2, 1, 0))
        xp1 = np.array([self.nelx/6, self.nelx/2, self.nelx*5/6])
        xp2 = np.array([0, self.nelx/3, self.nelx*2/3, self.nelx])
        xp = np.concatenate((xp1, xp2, xp1, xp2, xp1))
        yp = []
        for i in range(5):
            if i % 2:
                yp.extend([i/4*self.nely, i/4*self.nely, i/4*self.nely, i/4*self.nely])
            else:
                yp.extend([i/4*self.nely, i/4*self.nely, i/4*self.nely])
        yp = np.array(yp)
        for p in (xp, yp):
            p = np.array([p]).repeat(self.nely+1, axis=0)
            p = np.array([p]).repeat(self.nelx+1, axis=0)
        z = np.min(np.sqrt((x-xp)**2+(y-yp)**2), axis=2) - ((self.nelx/6)**2+(self.nely/4)**2)**0.5*0.25
        z = np.minimum(z, z1)
        return z.T/(((self.nelx/6)**2+(self.nely/4)**2)**0.5*0.25)


if __name__ == '__main__':
    sur = Surface(80, 40)
    # print(sur.matrixA())
    X = range(sur.nelx + 1)
    Y = range(sur.nely + 1)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, sur.origin(), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.show()
