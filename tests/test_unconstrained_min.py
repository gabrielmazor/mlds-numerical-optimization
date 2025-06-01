import numpy as np
import unittest
from .examples import quadratic_1, quadratic_2, quadratic_3, rosenbrock, linear, exponential
from src.unconstrained_min import LineSearchMinimizer
from src.utils import plot_contour, plot_function_values

class TestUnconstrainedMinimizer(unittest.TestCase):

    def test_quadratic_1(self):
        newton = LineSearchMinimizer(method='nm')
        gradient = LineSearchMinimizer(method='gd')
        x0 = np.array([1.0, 1.0])
        
        _, _, newton_success, newton_path_x, newton_path_f = newton.minimize(quadratic_1, x0)
        _, _, gradent_success, gradient_path_x, gradient_path_f = gradient.minimize(quadratic_1, x0)   
        
        plot_contour(quadratic_1, f_name="Quadratic Function 1", paths=[newton_path_x, gradient_path_x], path_label=['Newton', 'Gradient Descent'])
        plot_function_values([newton_path_f, gradient_path_f], f_name="Quadratic Function 1", labels=['Newton', 'Gradient Descent'])

    def test_quadratic_2(self):
        newton = LineSearchMinimizer(method='nm')
        gradient = LineSearchMinimizer(method='gd')
        x0 = np.array([1.0, 1.0])
        
        _, _, newton_success, newton_path_x, newton_path_f = newton.minimize(quadratic_2, x0)
        _, _, gradent_success, gradient_path_x, gradient_path_f = gradient.minimize(quadratic_2, x0)   
        
        plot_contour(quadratic_2, f_name="Quadratic Function 2", paths=[newton_path_x, gradient_path_x], path_label=['Newton', 'Gradient Descent'])
        plot_function_values([newton_path_f, gradient_path_f], f_name="Quadratic Function 2", labels=['Newton', 'Gradient Descent'])

    def test_quadratic_3(self):
        newton = LineSearchMinimizer(method='nm')
        gradient = LineSearchMinimizer(method='gd')
        x0 = np.array([1.0, 1.0])
        
        _, _, newton_success, newton_path_x, newton_path_f = newton.minimize(quadratic_3, x0)
        _, _, gradent_success, gradient_path_x, gradient_path_f = gradient.minimize(quadratic_3, x0)   
        
        plot_contour(quadratic_3, f_name="Quadratic Function 3", paths=[newton_path_x, gradient_path_x], path_label=['Newton', 'Gradient Descent'])
        plot_function_values([newton_path_f, gradient_path_f], f_name="Quadratic Function 3", labels=['Newton', 'Gradient Descent'])

    def test_rosenbrock(self):
        newton = LineSearchMinimizer(method='nm')
        gradient = LineSearchMinimizer(method='gd')
        x0 = np.array([-1.0, 2.0])
        
        _, _, newton_success, newton_path_x, newton_path_f = newton.minimize(rosenbrock, x0)
        _, _, gradent_success, gradient_path_x, gradient_path_f = gradient.minimize(rosenbrock, x0)   
        
        plot_contour(rosenbrock, f_name="Rosenbrock Function", paths=[newton_path_x, gradient_path_x], path_label=['Newton', 'Gradient Descent'], xlim=(-1.5, 1.2), ylim=(-0.3, 2.1))
        plot_function_values([newton_path_f, gradient_path_f], f_name="Rosenbrock Function", labels=['Newton', 'Gradient Descent'])

    def test_linear(self):
        gradient = LineSearchMinimizer(method='gd')
        x0 = np.array([1.0, 1.0])
        
        _, _, gradent_success, gradient_path_x, gradient_path_f = gradient.minimize(linear, x0)   
        
        plot_contour(linear, f_name="Linear Function", paths=[gradient_path_x], path_label=['Gradient Descent'], xlim=(-200, 2), ylim=(-600, 2))
        plot_function_values([gradient_path_f], f_name="Linear Function", labels=['Gradient Descent'])

    def test_exponential(self):
        newton = LineSearchMinimizer(method='nm')
        gradient = LineSearchMinimizer(method='gd')
        x0 = np.array([1.0, 1.0])
        
        _, _, newton_success, newton_path_x, newton_path_f = newton.minimize(exponential, x0)
        _, _, gradent_success, gradient_path_x, gradient_path_f = gradient.minimize(exponential, x0)   
        
        plot_contour(exponential, f_name="Exponential Function", paths=[newton_path_x, gradient_path_x], path_label=['Newton', 'Gradient Descent'])
        plot_function_values([newton_path_f, gradient_path_f], f_name="Exponential Function", labels=['Newton', 'Gradient Descent'])

if __name__ == '__main__':
    unittest.main()