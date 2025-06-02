import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
 
def plot_contour(f, f_name = None, paths = None, path_label = None, xlim=(-2, 2), ylim=(-2, 2), ax=None):
    x = np.linspace(xlim[0], xlim[1], 200)
    y = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            val, _, _ = f(np.array([X[i, j], Y[i, j]]), False)
            Z[i, j] = val
    
    if ax is None:
        fig, ax = plt.subplots()
    
    valid_z = Z[np.isfinite(Z)]
    
    if len(valid_z) > 0:
        z_min, z_max = np.min(valid_z), np.max(valid_z)
        
        # Different levels for different function types
        if f_name:
            if 'exponential' in f_name.split()[0].lower():
                # Use log scale for exponential functions
                z_min = max(z_min, 1e-10)  # Avoid log(0)
                levels = np.logspace(np.log10(z_min), np.log10(z_max), 15)
            else:
                # For quadratic and Rosenbrock, use more levels near minimum
                if z_min >= 0:
                    levels = np.logspace(np.log10(max(z_min + 1e-10, 1e-10)), 
                                    np.log10(z_max + 1), 10)
                else:
                    levels = np.linspace(z_min, z_max, 10)
        else:
            levels = 20

    cs = ax.contour(X, Y, Z, levels=levels, cmap='terrain', alpha=0.7)
    ax.clabel(cs)
    colors = ['green', 'orange']

    if paths is not None:
        for i, path in enumerate(paths):
            ax.plot(path[:,0], path[:,1], marker='o', linestyle='-', label=path_label[i] if path_label else None, alpha=0.7, color=colors[i])
    
    ax.set_title("Line Search Minimization")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()

def plot_function_values(f_vals, f_name=None, labels=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    colors = ['green', 'orange']
    for i, vals in enumerate(f_vals):
        ax.plot(range(len(vals)), vals, marker='o', linestyle='-', alpha=0.7, label = labels[i] if labels else None, color=colors[i])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Function Value")
    ax.legend()
    ax.set_title("Function Values")


def plot_both(f, f_name, paths_x, paths_f, xlim=None, ylim=None):
    if xlim is None:
        xlim = (-2, 2)
    if ylim is None:
        ylim = (-2, 2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
    plot_contour(f, f_name=f_name, paths=[paths_x[0], paths_x[1]], path_label=['Newton', 'Gradient Descent'], ax=ax1, xlim=xlim, ylim=ylim)
    plot_function_values([paths_f[0], paths_f[1]], f_name=f_name, labels=['Newton', 'Gradient Descent'], ax=ax2)

    plt.tight_layout()
    f_name = f_name.replace(" ", "_").lower()
    plt.savefig(f"plots/{f_name}.png", bbox_inches='tight', dpi = 300)
    plt.show()