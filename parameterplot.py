import matplotlib.pyplot as plt
import numpy as np
import torch

def add_contour(ax, x, y, z):
    cs = ax.contourf(x, y, np.log(z))
    plt.colorbar(cs)
    #ax.contour(cs, colors='k')
    #ax.colorbar(shrink=.92)

def add_trajectory(ax, x,y,b2):
    ax.plot(x[0], y[0], marker='o', label=b2[0])
    ax.plot(x[1], y[1], marker='o', label=b2[1])

def plot_all(dirname):

    surface = np.loadtxt(dirname + 'surface.txt')
    x = np.loadtxt(dirname + 'x.txt')
    y = np.loadtxt(dirname + 'y.txt')

    xn = np.loadtxt(dirname + 'x_grid.txt')
    yn = np.loadtxt(dirname + 'y_grid.txt')
    b2 = np.loadtxt(dirname + 'betatwo.txt')
    #print(y)
    fig, axs = plt.subplots()
    #xn = np.linspace(-1,1,surface.shape[0])
    #yn = np.linspace(-1,1,surface.shape[1])
    add_contour(axs, xn, yn, surface)
    add_trajectory(axs, x, y, b2)
    #plt.colorbar(ax=axs)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_all(r"D:\shi\projects\adam\raw_data\2020-09-24-16\\")