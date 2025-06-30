import unittest

from src.constrained_min import interior_pt
from src.utils import plot_constrained_optimization
from .examples import lp_objective, lp_ineq_constraints, lp_eq_mat, lp_eq_rhs, lp_x0
from .examples import qp_objective, qp_ineq_constraints, qp_eq_mat, qp_eq_rhs, qp_x0

class TestConstrainedMinimizer(unittest.TestCase):

    def test_qp(self):
        x, fx, success, path_x, path_f = interior_pt(qp_objective, qp_ineq_constraints, qp_eq_mat, qp_eq_rhs, qp_x0)
        print("\nQuadratic Programming")
        print(f"Objective value: {fx:.3f}, Optimal point: {x}")
        plot_constrained_optimization(path_x, path_f, "Quadratic Programming")

    def test_lp(self):
        x, fx, success, path_x, path_f = interior_pt(lp_objective, lp_ineq_constraints, lp_eq_mat, lp_eq_rhs, lp_x0)
        print("\nLinear Programming")
        print(f"Objective value: {-fx:.3f}, Optimal point: {x}")
        plot_constrained_optimization(path_x, -path_f, "Linear Programming")
    
if __name__ == '__main__':
    unittest.main()