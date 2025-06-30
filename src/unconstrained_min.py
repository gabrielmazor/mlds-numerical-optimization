import numpy as np

def backtracking(f, x, pk):
    alpha = 1
    c = 0.01
    rho = 0.5
    fx, gx, _ = f(x, False)
    fxalpha, _, _= f(x + alpha * pk, False)
    while fxalpha > fx + c * gx.T @ pk * alpha:
        alpha *= rho
        fxalpha, _, _= f(x + alpha * pk, False)
    return alpha

class LineSearchMinimizer:
    def __init__(self, method = 'gd'):
        self.method = method

    def minimize(self, f, x0, obj_tol = 1e-12, param_tol = 1e-8, max_iter = 100):
        # initialize
        x = x0.copy()
        path_x = []
        path_f = []
        success = False
        
        for i in range(max_iter):
            if self.method == 'gd':
                fx, gx, _ = f(x, False)
                pk = -gx
            
            elif self.method == "nm":
                fx, gx, bk = f(x, True)
                if np.linalg.det(bk) == 0:
                    raise ValueError("Hessian is singular, cannot compute Newton step.")

                pk = np.linalg.solve(bk, -gx)
            
            path_x.append(x.copy())
            path_f.append(fx)
            
            alpha = backtracking(f, x, pk)
            prt = f"Iteration {i}: x = {x}, f(x) = {fx}, a = {alpha}"
            print(prt)
            x += alpha * pk

            # Termination conditions
            if len(path_f) > 1:
                if np.linalg.norm(path_f[-2] - fx) < obj_tol:
                    success = True
                    break
                if np.linalg.norm(path_x[-2] - x) < param_tol:
                    success = True
                    break
            if np.linalg.norm(gx) < param_tol:
                success = True
                break
            if self.method =='nm':
                if 0.5 * pk.T @ bk @ pk < obj_tol:
                    success = True
                    break

        return x, fx, success, prt, np.array(path_x), np.array(path_f)