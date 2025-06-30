import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon
 
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


def plot_constrained_optimization(x_path, f_path, f_name=""):
    n_dims = x_path.shape[1]
    
    fig = plt.figure(figsize=(16,8))
    
    # First plot - Feasble region and central path
    if n_dims == 2:
        # LP
        ax1 = fig.add_subplot(121)
        
        # LP example feasible region 
        vertices = np.array([[0, 1], [1, 0], [2, 0], [2, 1]])
        polygon = Polygon(vertices, alpha=0.3, facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
        ax1.add_patch(polygon)
        
        # Plot constraint lines
        x_range = np.linspace(-0.5, 2.5, 100)
        ax1.plot(x_range, -x_range + 1, 'b-', linewidth=2, label='y = -x + 1')
        ax1.axhline(y=1, color='g', linewidth=2, label='y = 1')
        ax1.axvline(x=2, color='r', linewidth=2, label='x = 2')
        ax1.axhline(y=0, color='m', linewidth=2, label='y = 0')
        
        ax1.set_xlim(-0.5, 2.5)
        ax1.set_ylim(-0.5, 1.5)
        
        # Plot path
        ax1.plot(x_path[:, 0], x_path[:, 1], 'ko-', linewidth=3, markersize=8,label='Central Path', markerfacecolor='yellow')
        ax1.scatter(x_path[0, 0], x_path[0, 1], color='green', s=150, marker='o', label='Initial', edgecolor='darkgreen', linewidth=2)
        ax1.scatter(x_path[-1, 0], x_path[-1, 1], color='red', s=200, marker='*', label='Solution', edgecolor='darkred', linewidth=2)
        
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('y', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
    elif n_dims == 3:
        # QP
        ax1 = fig.add_subplot(121, projection='3d')
        
        # QP example feasible region 
        vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        faces = [[vertices[0], vertices[1], vertices[2]]]
        triangle = Poly3DCollection(faces, alpha=0.3, facecolor='lightblue', edgecolor='navy', linewidth=2)
        ax1.add_collection3d(triangle)
        
        # Feasible edges
        for i in range(3):
            for j in range(i+1, 3):
                ax1.plot([vertices[i][0], vertices[j][0]], 
                        [vertices[i][1], vertices[j][1]], 
                        [vertices[i][2], vertices[j][2]], 'navy', linewidth=2)
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_zlim(0, 1)
        
        # Plot path
        ax1.plot(x_path[:, 0], x_path[:, 1], x_path[:, 2], 'ko-', linewidth=3, markersize=8, label='Central Path', markerfacecolor='yellow')
        ax1.scatter(*x_path[0], color='green', s=150, marker='o', label='Initial', edgecolor='darkgreen', linewidth=2)
        ax1.scatter(*x_path[-1], color='red', s=200, marker='*', label='Solution', edgecolor='darkred', linewidth=2)
        
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('y', fontsize=12)
        ax1.set_zlabel('z', fontsize=12)
        ax1.legend()
        ax1.view_init(elev=20, azim=45)
    
    ax1.set_title(f'Feasible Region and Central Path', fontsize=14)
    
    # Second plot - objective over iterations
    ax2 = fig.add_subplot(122)
    ax2.plot(range(len(f_path)), f_path, 'b-o', linewidth=3, markersize=8,markerfacecolor='lightblue', markeredgecolor='darkblue')
    ax2.set_xlabel('Outer Iteration Number', fontsize=12)
    ax2.set_ylabel('Objective Value', fontsize=12)
    ax2.set_title(f'Objective Convergence', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'{f_name} Constrained Optimization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    f_name = f_name.replace(" ", "_").lower()
    plt.savefig(f"plots/{f_name}.png", bbox_inches='tight', dpi = 300)
    plt.show()