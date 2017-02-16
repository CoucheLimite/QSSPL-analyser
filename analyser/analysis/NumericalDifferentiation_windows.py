
#-------------------------------------------------------------------------
# Name:        Numerical Differentiation
# Purpose:     To Differentiate
#
# Author:      Mattias
#
# Created:     26/06/2013
# Copyright:   (c) Mattias 2012
# Licence:     <my licence>
#-------------------------------------------------------------------------
#!/usr/bin/env pytho
#import numpy as np
import numpy as np

import scipy.sparse as SS
from scipy.sparse.linalg import spsolve
#from matplotlib import  *


def TwoPointCentral(x, y):
    # 2-point formula
    dyf = np.zeros(x.shape[0])
    for i in range(y.shape[0] - 1):
        dyf[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    # set last element by backwards difference
    dyf[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    #plot(x,dyf,'r-',label='2pt-forward diff')
    # semilogy()
    # show()
    return dyf


def FourPointCentral(x, y):
    '''
    Assumes evenly spaced points!!

    calculate dy by 4-point center differencing using array slices

    \frac{y[i-2] - 8y[i-1] + 8[i+1] - y[i+2]}{12h}

    y[0] and y[1] must be defined by lower order methods
    and y[-1] and y[-2] must be defined by lower order methods
    '''

    dy = np.zeros(y.shape, np.float)  # we know it will be this size
    h = x[1] - x[0]  # this assumes the points are evenely spaced!
    dy[2:-2] = (y[0:-4] - 8 * y[1:-3] + 8 * y[3:-1] - y[4:]) / (12. * h)

    dy[0] = (y[1] - y[0]) / (x[1] - x[0])
    dy[1] = (y[2] - y[1]) / (x[2] - x[1])
    dy[-2] = (y[-2] - y[-3]) / (x[-2] - x[-3])
    dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return dy


def FirstDerivative(X, Y, lam):
    """Provides the first derivitave of something, WITH EQUALLY SPACED POINTS"""

    Xreal = np.matrix(X).T
    Yreal = _Derivitivematrix(
        1, Xreal[1] - Xreal[0], np.matrix(Y).T.shape[0]) * np.matrix(Y).T

    # obtaining Differential np.matrix
    D = _Derivitivematrix(2, Xreal[1] - Xreal[0], Yreal.shape[0])

    # usually would lose a point, as evaultes inbetween points, so conver
    # back to total number, with error at the end areas
    return np.interp(X, (X[:-1] + X[1:]) / 2, self.Smoothed(lam, Yreal, D))


def _Derivitivematrix(Order, Deltax, n):

    # Have to use SS as numpy can't deal with these big arrays.
    # Also its a smaller demand on the system as it only records/does
    # opperations of values with number
    D = -SS.eye(n - 1, n) + SS.eye(n - 1, n, 1)

    # Computing the correct Order (the order is the differential to make smooth) np.matrix
    # It should be 2 orders higher than the higest differential you are
    # going to use
    for i in range(1, Order):
        D = D[:-1, :-i] * D

    D = (Deltax[0, 0]**(-Order)) * D

    return D


def Smoothed(self, lam, Y, D):

    return spsolve((SS.identity(D.T.shape[0]) + lam * D.T * D), Y).T[:, 0]


if __name__ == '__main__':
    # Comparing
    import matplotlib.pylab as plt

    X = np.linspace(-10, 10, 10000)

    Y = -X**5 + 4 * X**4 + 5e1 * X**3 - 1e2 * X**2 + 5e2 * X + 1e4
    Yd = -5 * X**4 + 16 * X**3 + 1.5e2 * X**2 - 2e2 * X + 5e2

    #Y = exp(X/0.02585)
    #Yd = exp(X/0.02585)/0.02585

    # plt.plot(X,Regularisation().FirstDerivative(X,Y,0),'b--',label='Regularisation Derivative')
    plt.plot(X, Yd, 'b-', label='Real derivative')
    plt.plot(X, TwoPointCentral(X, Y),
             'r--', label='Two Point Finite Difference')
    plt.plot(X, FourPointCentral(X, Y),
             'g-.', label='Four Point Finite Difference')
    # grid(True,'major')
    plt.legend(loc=2)
    # semilogy()
    # twinx()
    #plot(X,Y,'k,',label='Original function')
    # legend(loc=1)
    # semilogy()

    plt.show()
