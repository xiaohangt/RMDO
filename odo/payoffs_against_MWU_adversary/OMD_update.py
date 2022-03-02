
#Online Gradient Descent with lazy projections
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds

def OMD(eta, theta, n,x0):
    linear_constraint=LinearConstraint(np.transpose(np.ones(n)),np.array([1]),np.array([1]))
    bounds=Bounds(np.zeros(n), np.ones(n))
    def problem(x):
        vector=x-eta*theta
        return sum(vector[:]**2)
    def problem_gradient(x):
        return 2*(x-eta*theta)
    def problem_hes(x):
        return 2* np.identity(n)
    res=minimize(problem, x0,method='trust-constr', jac=problem_gradient,hess=problem_hes, constraints=[linear_constraint],options={'verbose':1}, bounds=bounds)
    return res.x
    