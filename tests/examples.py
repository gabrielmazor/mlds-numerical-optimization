import numpy as np

# Unconstrained examples
def quadratic_1(x, hessian = False):
    Q = np.array([[1,0], [0,1]])
    f = x.T @ Q @ x
    g = 2 * Q @ x
    h = Q if hessian else None

    return f, g, h

def quadratic_2(x, hessian = False):
    Q = np.array([[1,0], [0,100]])
    f = x.T @ Q @ x
    g = 2 * Q @ x
    h = Q if hessian else None

    return f, g, h

def quadratic_3(x, hessian = False):
    Q1 = np.array([[np.sqrt(3)/2,-0.5], [-0.5,np.sqrt(3)/2]])
    Q2 = np.array([[100, 0], [0, 1]])
    Q = Q1.T @ Q2 @ Q1
    f = x.T @ Q @ x
    g = 2 * Q @ x
    h = Q if hessian else None

    return f, g, h

def rosenbrock(x, hessian = False):
    f = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    df_dx0 = -400 * x[0] * (x[1] - x[0]**2) + 2 * (x[0] - 1)
    df_dx1 = 200 * (x[1] - x[0]**2)
    g = np.array([df_dx0, df_dx1])
    if hessian:
        d2f_dx0dx0 = -400 * (x[1] - x[0]**2) + 800 * x[0]**2 + 2
        d2f_dx0dx1 = -400 * x[0]
        d2f_dx1dx1 = 200
        h = np.array([[d2f_dx0dx0, d2f_dx0dx1], [d2f_dx0dx1, d2f_dx1dx1]])
    else:
        h = None
    
    return f, g, h

def linear(x, hessian = False):
    a = np.array([2.0, 5.5])
    f = a.T @ x
    g = a
    h = np.zeros((2, 2)) if hessian else None

    return f, g, h

def exponential(x, hessian = False):
    f = np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) + np.exp(-x[0] - 0.1)
    df_dx0 = np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) - np.exp(-x[0] - 0.1)
    df_dx1 = 3 * np.exp(x[0] + 3*x[1] - 0.1) - 3 * np.exp(x[0] - 3*x[1] - 0.1)
    g = np.array([df_dx0, df_dx1])
    if hessian:
        d2f_dx0dx0 = np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) + np.exp(-x[0] - 0.1)
        d2f_dx0dx1 = 3 * np.exp(x[0] + 3*x[1] - 0.1) - 3 * np.exp(x[0] - 3*x[1] - 0.1)
        d2f_dx1dx1 = 9 * np.exp(x[0] + 3*x[1] - 0.1) + 9 * np.exp(x[0] - 3*x[1] - 0.1)
        h = np.array([[d2f_dx0dx0, d2f_dx0dx1], [d2f_dx0dx1, d2f_dx1dx1]])
    else:
        h = None
    
    return f, g, h

# Constrained examples
# Linear programming example
def lp_objective(x, hessian = False):
    f = x[0] + x[1]
    g = np.array([1.0, 1.0])
    h = np.zeros((2, 2)) if hessian else None

    return -f, -g, h

# Inequality constraints
def lp_halfplane(x, hessian = False):
    f = -x[0] - x[1] + 1
    g = np.array([-1.0, -1.0])
    h = np.zeros((2, 2)) if hessian else None

    return f, g, h

def lp_y_ineq(x, hessian = False):
    f = x[1] - 1
    g = np.array([0.0, 1.0])
    h = np.zeros((2, 2)) if hessian else None

    return f, g, h

def lp_y_nonneg(x, hessian = False):
    y = - x[1]
    g = np.array([0.0, -1.0])
    h = np.zeros((2, 2)) if hessian else None

    return y, g, h

def lp_x_ineq(x, hessian = False):
    f = x[0] - 2
    g = np.array([1.0, 0.0])
    h = np.zeros((2, 2)) if hessian else None

    return f, g, h

lp_ineq_constraints = [lp_halfplane, lp_y_ineq, lp_y_nonneg, lp_x_ineq]

# Equality constraints
lp_eq_mat = None
lp_eq_rhs = None

# Initial point
lp_x0 = np.array([0.5, 0.75])

# Quadratic programming example
def qp_objective(x, hessian = False):
    f = x[0]**2 + x[1]**2 +(x[2]+1)**2
    g = np.array([2*x[0], 2*x[1], 2*(x[2]+1)])
    h = np.diag([2.0, 2.0, 2.0]) if hessian else None

    return f, g, h

# Inquality constraints
def qp_x_nonneg(x, hessian = False):
    x = -x[0]
    g = np.array([-1.0, 0.0, 0.0])
    h = np.zeros((3, 3)) if hessian else None

    return x, g, h

def qp_y_nonneg(x, hessian = False):
    y = -x[1]
    g = np.array([0.0, -1.0, 0.0])
    h = np.zeros((3, 3)) if hessian else None

    return y, g, h

def qp_z_nonneg(x, hessian = False):
    z = -x[2]
    g = np.array([0.0, 0.0, -1.0])
    h = np.zeros((3, 3)) if hessian else None

    return z, g, h

qp_ineq_constraints = [qp_x_nonneg, qp_y_nonneg, qp_z_nonneg]

# Equality constraints
qp_eq_mat = np.array([[1.0, 1.0, 1.0]])
qp_eq_rhs = np.array([1.0])

# Initial point
qp_x0 = np.array([0.1, 0.2, 0.7])