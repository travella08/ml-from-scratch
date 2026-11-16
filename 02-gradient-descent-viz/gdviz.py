import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt


# --- Contour plots -------------------------------------------------------

def plot_contours(surface, ax, n_levels=30, log_levels=False, resolution=300):
    xs = np.linspace(*surface.xlim, resolution)
    ys = np.linspace(*surface.ylim, resolution)
    X, Y = np.meshgrid(xs, ys)    
    Z = surface.f(X, Y)                             
    if log_levels:
        levels = np.logspace(-2, np.log10(Z.max()), n_levels)
    else:
        levels = n_levels
    ax.contour(X, Y, Z, levels=levels, cmap="viridis", linewidths=0.8)
    for m in surface.minima:
        ax.plot(*m, "r*", markersize=12)
    ax.set_aspect("equal")
    ax.set_title(surface.name)




@dataclass
class Surface:
    name: str
    f: object        # f(x, y) -> same shape as x
    grad: object     # grad(p), p shape (2,) -> shape (2,)
    xlim: tuple
    ylim: tuple
    minima: list     # [(x, y), ...]


# --- Round bowl: f(x, y) = x^2 + y^2 -------------------------------------

def round_bowl_f(x, y):
    """Value of the round bowl.

    Inputs:  x, y: floats or arrays of the same shape (e.g. meshgrid output).
    Output:  same shape as x.
    """
    return x**2 + y**2


def round_bowl_grad(p):
    """Gradient of the round bowl at one point.

    Input:   p: array of shape (2,), the point (x, y).
    Output:  array of shape (2,): (df/dx, df/dy).
    """
    return 2*p

# --- Stretched bowl: f(x, y) = x^2 + 10 y^2 ------------------------------

def stretched_bowl_f(x, y):
    """Value of the stretched bowl.

    Inputs:  x, y: floats or arrays of the same shape.
    Output:  same shape as x.
    """
    return x**2 + 10 * y**2


def stretched_bowl_grad(p):
    """Gradient of the stretched bowl at one point.

    Input:   p: array of shape (2,).
    Output:  array of shape (2,).
    """
    return 2*np.array([1,10])*p

# --- Rosenbrock: f(x, y) = (1 - x)^2 + 100 (y - x^2)^2 -------------------

def rosenbrock_f(x, y):
    """Value of the Rosenbrock function. Minimum f = 0 at (1, 1).

    Inputs:  x, y: floats or arrays of the same shape.
    Output:  same shape as x.
    """
    return (1-x)**2 + 100*(y - x**2)**2

def rosenbrock_grad(p):
    """Gradient of the Rosenbrock function at one point.

    Input:   p: array of shape (2,).
    Output:  array of shape (2,).
    """
    x, y = p
    partial_x = -2*(1-x) - 400*x*(y-x**2)
    partial_y = 200*(y-x**2)
    return np.array([partial_x, partial_y])

# --- Himmelblau: f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2 ------------

def himmelblau_f(x, y):
    """Value of Himmelblau's function. Four minima, all f = 0.

    Inputs:  x, y: floats or arrays of the same shape.
    Output:  same shape as x.
    """
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def himmelblau_grad(p):
    """Gradient of Himmelblau's function at one point.

    Input:   p: array of shape (2,).
    Output:  array of shape (2,).
    """
    x, y = p
    partial_x = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
    partial_y = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
    return np.array([partial_x, partial_y])
    


# --- Gradient checking ---------------------------------------------------

def numerical_grad(surface, p, h=1e-5):
    """Central-difference estimate of the gradient of surface.f at p.

    Inputs:  surface: a Surface.
             p: array of shape (2,).
             h: small step size.
    Output:  array of shape (2,).
    """
    partial_x = (surface.f(*p+[h, 0])-surface.f(*p-[h, 0])) / (2*h)
    partial_y = (surface.f(*p+[0, h])-surface.f(*p-[0, h])) / (2*h)
    return np.array([partial_x, partial_y])


def check_grad(surface, n_points=20, seed=0):
    """Compare surface.grad with numerical_grad at random points.

    Inputs:  surface: a Surface.
             n_points: how many random points inside the plot window to test.
             seed: random seed so the check is repeatable.
    Output:  float: the largest relative error found (should be below 1e-6).
    """
    largest_rel_error = 0.0
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(n_points):
        x = rng.uniform(surface.xlim[0], surface.xlim[1])
        y = rng.uniform(surface.ylim[0], surface.ylim[1])
        points.append(np.array([x, y]))
    points = np.array(points)
    for p in points:
        analytic = surface.grad(p)
        numerical = numerical_grad(surface, p, h=1e-5)
        rel_error = np.linalg.norm(analytic - numerical) / (np.linalg.norm(analytic) + np.linalg.norm(numerical) + 1e-12)
        largest_rel_error = max(largest_rel_error, rel_error)
    return largest_rel_error

# --- Surface instances (fill in after the functions above work) ----------

ROUND_BOWL = Surface(
    name="round bowl",
    f=round_bowl_f,
    grad=round_bowl_grad,
    xlim=(-5,5),
    ylim=(-5,5),
    minima=[(0,0)]
)
STRETCHED_BOWL = Surface(
    name="stretched bowl",
    f=stretched_bowl_f,
    grad=stretched_bowl_grad,
    xlim=(-5,5),
    ylim=(-5,5),
    minima=[(0,0)]
)
ROSENBROCK = Surface(
    name="rosenbrock surface",
    f=rosenbrock_f,
    grad=rosenbrock_grad,
    xlim=(-2, 2),
    ylim=(-1,3),
    minima=[(1,1)]
)
HIMMELBLAU = Surface(
    name="himmelblau surface",
    f=himmelblau_f,
    grad=himmelblau_grad,
    xlim=(-5, 5),
    ylim=(-5,5),
    minima=[(3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)]
)
