import numpy as np
import unittest
from .examples import quadratic_1, quadratic_2, quadratic_3, rosenbrock, linear, exponential
from src.unconstrained_min import LineSearchMinimizer
from src.utils import plot_contour, plot_function_values, plot_both

class TestUnconstrainedMinimizer(unittest.TestCase):

    def test_quadratic_1(self):
        newton = LineSearchMinimizer(method='nm')
        gradient = LineSearchMinimizer(method='gd')
        x0 = np.array([1.0, 1.0])
        
        _, _, newton_success, prt, newton_path_x, newton_path_f = newton.minimize(quadratic_1, x0)
        _, _, gradent_success, prt, gradient_path_x, gradient_path_f = gradient.minimize(quadratic_1, x0)   
        
        plot_both(quadratic_1, "Quadratic Function 1", [newton_path_x, gradient_path_x],[newton_path_f, gradient_path_f])

    def test_quadratic_2(self):
        newton = LineSearchMinimizer(method='nm')
        gradient = LineSearchMinimizer(method='gd')
        x0 = np.array([1.0, 1.0])
        
        _, _, newton_success, prt, newton_path_x, newton_path_f = newton.minimize(quadratic_2, x0)
        _, _, gradent_success, prt, gradient_path_x, gradient_path_f = gradient.minimize(quadratic_2, x0)   
        
        plot_both(quadratic_2, "Quadratic Function 2", [newton_path_x, gradient_path_x],[newton_path_f, gradient_path_f])

    def test_quadratic_3(self):
        newton = LineSearchMinimizer(method='nm')
        gradient = LineSearchMinimizer(method='gd')
        x0 = np.array([1.0, 1.0])
        
        _, _, newton_success, prt, newton_path_x, newton_path_f = newton.minimize(quadratic_3, x0)
        _, _, gradent_success, prt, gradient_path_x, gradient_path_f = gradient.minimize(quadratic_3, x0)   
        
        plot_both(quadratic_3, "Quadratic Function 3", [newton_path_x, gradient_path_x],[newton_path_f, gradient_path_f])

    def test_rosenbrock(self):
        newton = LineSearchMinimizer(method='nm')
        gradient = LineSearchMinimizer(method='gd')
        x0 = np.array([-1.0, 2.0])
        
        _, _, newton_success, prt, newton_path_x, newton_path_f = newton.minimize(rosenbrock, x0)
        _, _, gradent_success, prt, gradient_path_x, gradient_path_f = gradient.minimize(rosenbrock, x0)   
        
        plot_both(rosenbrock, "Rosenbrock Function", [newton_path_x, gradient_path_x],[newton_path_f, gradient_path_f], xlim=(-1.5, 1.2), ylim=(-0.3, 2.1))

    def test_linear(self):
        gradient = LineSearchMinimizer(method='gd')
        x0 = np.array([1.0, 1.0])
        
        _, _, gradent_success, prt, gradient_path_x, gradient_path_f = gradient.minimize(linear, x0)   
        
        plot_both(linear, "Linear Function", [gradient_path_x, gradient_path_x],[gradient_path_f, gradient_path_f], xlim=(-200, 2), ylim=(-600, 2))

    def test_exponential(self):
        newton = LineSearchMinimizer(method='nm')
        gradient = LineSearchMinimizer(method='gd')
        x0 = np.array([1.0, 1.0])
        
        _, _, newton_success, prt, newton_path_x, newton_path_f = newton.minimize(exponential, x0)
        _, _, gradent_success, prt, gradient_path_x, gradient_path_f = gradient.minimize(exponential, x0)   
        
        plot_both(exponential, "Exponential Function", [newton_path_x, gradient_path_x],[newton_path_f, gradient_path_f])

if __name__ == '__main__':
    unittest.main()