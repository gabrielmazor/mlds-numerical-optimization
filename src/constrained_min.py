import numpy as np
from numpy.linalg import norm, solve

from .unconstrained_min import LineSearchMinimizer, backtracking


def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
    # Parameters
    t = 1.0 
    mu = 10.0 
    eps_tol = 1e-6 
    max_outer_iter = 50
    max_inner_iter = 100
    
    # Initialize
    x = x0.copy()
    m = len(ineq_constraints)
    has_eq = eq_constraints_mat is not None and eq_constraints_mat.size > 0
    
    path_x = [x0.copy()]
    path_f = []
    success = False
    
    fx, _, _ = func(x0, False)
    path_f.append(fx)
    
    # Outer iterations
    for outer_iter in range(max_outer_iter):
        # get the barrier function for current t
        barrier = get_barrier(func, t, ineq_constraints)
        
        # Solve the barrier iteration with kkt or unconstrained min
        if has_eq:
            x_new = newton_eq_constrained(barrier, x, eq_constraints_mat, eq_constraints_rhs, max_iter=max_inner_iter)
        else:
            minimizer = LineSearchMinimizer(method='nm')
            x_new, _, _, _, _, _ = minimizer.minimize(barrier, x, obj_tol=1e-10, param_tol=1e-8, max_iter=max_inner_iter)
        
        x = x_new

        # store path
        fx, _, _ = func(x, False)
        path_x.append(x.copy())
        path_f.append(fx)
        
        # duality gap
        gap = m / t
        
        # Check early termination
        if gap < eps_tol:
            success = True
            return x, fx, success, np.array(path_x), np.array(path_f)
        
        # Increase t
        t *= mu
    
    return x, fx, success, np.array(path_x), np.array(path_f)


def newton_eq_constrained(func, x0, A, b, max_iter=100, tol=1e-8):
    # initialize
    x = x0.copy()
    n = len(x)
    m = A.shape[0] 
    
    for i in range(max_iter):
        fx, gx, Hx = func(x, True)
        
        residual = A @ x - b
        
        # Form KKT system
        kkt_mat = np.zeros((n + m, n + m))
        kkt_mat[:n, :n] = Hx
        kkt_mat[:n, n:] = A.T
        kkt_mat[n:, :n] = A
        
        kkt_rhs = np.zeros(n + m)
        kkt_rhs[:n] = -gx
        kkt_rhs[n:] = -residual
        
        solution = solve(kkt_mat, kkt_rhs)
        dx = solution[:n]
        
        alpha = backtracking(func, x, dx)
        x = x + alpha * dx
        
        # Check early termination
        if norm(dx) < tol and norm(residual) < tol:
            return x
    
    return x

def get_barrier(func, t, ineq_constraints):
    def log_barrier(z, need_hessian):
        f_val, f_grad, f_hess = func(z, need_hessian)
        
        barrier_val = 0
        barrier_grad = np.zeros_like(z)
        if need_hessian:
            barrier_hess = np.zeros((len(z), len(z)))
        else:
            barrier_hess = None
        
        for g in ineq_constraints:
            g_val, g_grad, g_hess = g(z, need_hessian)
            
            if g_val >= 0:
                # infeasible region
                return np.inf, np.inf * np.ones_like(z), np.inf * np.ones((len(z), len(z)))
            
            barrier_val -= np.log(-g_val)
            barrier_grad += g_grad / (-g_val)
            
            if need_hessian:
                barrier_hess += np.outer(g_grad, g_grad) / (g_val**2)
                barrier_hess -= g_hess / (-g_val)
        
        # compute total
        total_val = t * f_val + barrier_val
        total_grad = t * f_grad + barrier_grad
        
        if need_hessian:
            total_hess = t * f_hess + barrier_hess
        else:
            total_hess = None
            
        return total_val, total_grad, total_hess
    return log_barrier