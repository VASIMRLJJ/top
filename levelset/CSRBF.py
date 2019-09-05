import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt


class CSRBF:
    def __init__(self, dml, xi, yi):
        self.dml = dml
        self.xi = xi
        self.yi = yi

    def c2(self, x, y):
        x, y = np.meshgrid(x, y)
        dl = np.sqrt((x-self.xi)**2 + (y-self.yi)**2)
        r = dl/self.dml
        fai = np.maximum(1-r, np.zeros(r.shape))**4*(4*r + 1)
        return fai

    def c4(self, x, y):
        x, y = np.meshgrid(x, y)
        dl = np.sqrt((x-self.xi)**2 + (y-self.yi)**2)
        r = dl/self.dml
        fai = np.maximum(1-r, np.zeros(r.shape))**6*(35*r**2 + 18*r + 3)
        return fai

    def c6(self, x, y):
        x, y = np.meshgrid(x, y)
        dl = np.sqrt((x-self.xi)**2 + (y-self.yi)**2)
        r = dl/self.dml
        fai = np.maximum(1-r, np.zeros(r.shape))**8*(32*r**3 + 8*r + 1)
        return fai


if __name__ == '__main__':
    r = 3.5
    X = np.arange(-1, r+1, 0.02)
    Y = np.arange(-1, r+1, 0.02)
    c0 = CSRBF(r, 0, 0)
    c1 = CSRBF(r, r, 0)
    c2 = CSRBF(r, r, r)
    c3 = CSRBF(r, 0, r)
    Z = c0.c2(X, Y)+c1.c2(X, Y)+c2.c2(X, Y)+c3.c2(X, Y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.show()
