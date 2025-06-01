import numpy as np

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