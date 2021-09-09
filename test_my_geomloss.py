import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

import torch
from mygeomloss import SamplesLoss as MySamplesLoss

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

import numpy as np
import torch
from random import choices
from imageio import imread
from matplotlib import pyplot as plt


def load_image(fname):
    img = imread(fname, as_gray=True)  # Grayscale
    img = (img[::-1, :]) / 255.0
    return 1 - img


def draw_samples(fname, n, dtype=torch.FloatTensor):
    A = load_image(fname)
    xg, yg = np.meshgrid(np.linspace(0, 1, A.shape[0]), np.linspace(0, 1, A.shape[1]))

    grid = list(zip(xg.ravel(), yg.ravel()))
    dens = A.ravel() / A.sum()
    dots = np.array(choices(grid, dens, k=n))
    dots += (0.5 / A.shape[0]) * np.random.standard_normal(dots.shape)

    return torch.from_numpy(dots).type(dtype)


def display_samples(ax, x, color):
    x_ = x.detach().cpu().numpy()
    ax.scatter(x_[:, 0], x_[:, 1], 25 * 500 / len(x_), color, edgecolors="none")

N, M = (1000, 1000) 

X_i = draw_samples("Heart.png", N, dtype)
Y_j = draw_samples("star.png", M, dtype)

def gradient_flow(loss, lr=0.05):
    """Flows along the gradient of the cost function, using a simple Euler scheme.

    Parameters:
        loss ((x_i,y_j) -> torch float number):
            Real-valued loss function.
        lr (float, default = .05):
            Learning rate, i.e. time step.
    """

    # Parameters for the gradient descent
    Nsteps = int(5 / lr) + 1
    display_its = [int(t / lr) for t in [0, 0.25, 0.50, 1.0, 2.0, 5.0]]

    # Use colors to identify the particles
    colors = (10 * X_i[:, 0]).cos() * (10 * X_i[:, 1]).cos()
    colors = colors.detach().cpu().numpy()

    # Make sure that we won't modify the reference samples
    x_i, y_j = X_i.clone(), Y_j.clone()

    # We're going to perform gradient descent on Loss(α, β)
    # wrt. the positions x_i of the diracs masses that make up α:
    x_i.requires_grad = True
    R = 5
    t_0 = time.time()
    plt.figure(figsize=(12, 8))
    k = 1
    for i in range(Nsteps):  # Euler scheme ===============
        # Compute cost and gradient
        L_αβ = loss(x_i, y_j, param=[R])
        [g] = torch.autograd.grad(L_αβ, [x_i])

        if i in display_its:  # display
            ax = plt.subplot(2, 3, k)
            k = k + 1
            plt.set_cmap("hsv")
            plt.scatter(
                [10], [10]
            )  # shameless hack to prevent a slight change of axis...

            display_samples(ax, y_j, [(0.55, 0.55, 0.95)])
            display_samples(ax, x_i, colors)

            ax.set_title("t = {:1.2f}".format(lr * i))

            plt.axis([0, 1, 0, 1])
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()

        # in-place modification of the tensor's values
        x_i.data -= lr * len(x_i) * g
    plt.title(
        "t = {:1.2f}, elapsed time: {:.2f}s/it".format(
            lr * i, (time.time() - t_0) / Nsteps
        )
    )


L = 1
mycost = '(SqDist(X, Y) + SqDist(X, X))'
mycost = '(Min(Concat(SqDist(Elem(X, 0) - IntCst({}), Elem(Y, 0)), Concat(SqDist(Elem(X, 0) + IntCst({}), Elem(Y, 0)), SqDist(Elem(X, 0), Elem(Y, 0)))) + SqDist(Elem(X, 1), Elem(Y, 1))) / IntCst(2))'.format(int(L), int(L))  # noqa E501
mycost = '(SqDist(X,Y) + SqDist(X, R))'

loss = MySamplesLoss(cost=mycost, params=['R = Pm(1)'], blur=0.5, backend='online')

# gradient_flow(loss, 0.01)
plt.savefig('default.png')

from pykeops.numpy import Genred

myfunc = Genred('X + Y', ['X = Pm(1)', 'Y = Pm(1)'],  reduction_op='Sum')
print("this is what i got", myfunc(3, 4))
