# common functions, such as plotting, printouts to console, etc.
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
 
def plot_contour(f, f_name = None, paths = None, path_label = None, xlim=(-2, 2), ylim=(-2, 2)):
    x = np.linspace(xlim[0], xlim[1], 200)
    y = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            val, _, _ = f(np.array([X[i, j], Y[i, j]]), False)
            Z[i, j] = val
    
    plt.figure(figsize=(8, 8))
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

    cs = plt.contour(X, Y, Z, levels=levels, cmap='terrain', alpha=0.7)
    plt.clabel(cs)

    if paths is not None:
        for i, path in enumerate(paths):
            plt.plot(path[:,0], path[:,1], marker='o', linestyle='-', label=path_label[i] if path_label else None, alpha=0.7)
    
    if f_name:
        title = f"Line Search Minimization for {f_name}"
        plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.savefig(f"plots/{title}.png", bbox_inches='tight', dpi = 300)
    plt.show()
    

def plot_function_values(f_vals, f_name=None, labels=None):
    plt.figure(figsize=(8, 5))
    for i, vals in enumerate(f_vals):
        plt.plot(range(len(vals)), vals, marker='o', linestyle='-', alpha=0.7, label = labels[i] if labels else None)
    plt.xlabel("Iteration")
    plt.ylabel("Function Value")
    plt.legend()
    if f_name:
        title = f"Function Values for {f_name}"
        plt.title(title)
    plt.savefig(f"plots/{title}.png", bbox_inches='tight', dpi = 300)
    plt.show()