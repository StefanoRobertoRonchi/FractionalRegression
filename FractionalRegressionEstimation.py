
import numpy as np
from scipy import sparse

class FracRegressionEstimation:
    def __init__(self, model):
        self.model = model



    def fit(self, X, y):
        # Define Backtracking line search function
        def backtracking_line_search(logl, X, y, Beta_prev, D, g,
                                alpha0=1.0, rho=0.5, c=1e-4, min_alpha=1e-8):
            direction = D @ g                          # descent direction
            grad_dot_dir = float(g.T @ direction)        # directional derivative (scalar)
            if grad_dot_dir <= 0:
                return 0.0                               # not a descent direction
            alpha = alpha0
            f0 = float(logl(X, y, Beta_prev))
            while alpha > min_alpha:
                Beta_cand = Beta_prev + alpha * direction
                f_cand = float(logl(X, y, Beta_cand))
                if f_cand >= f0 + c * alpha * grad_dot_dir:
                    return alpha
                alpha *= rho
            return alpha
        # Implement the fitting procedure for fractional regression
        # ======= Step 0 - Defiinition of Log Likelihood function =======
        X = sparse.csr_matrix(np.asmatrix(X))    
        y = np.asarray(y)
        lambda logl(X,y,*args): np.sum(y * (X @ Beta) - np.logaddexp(0,X @ Beta))
        lambda l_grad(X,y,*args): X.T @ (y - 1/(1+np.exp(-X @ Beta))) # Convex Function
        tol = 1e-5
        max_iter = 1000
        # Starting values for the optimization
        D_prev = np.eye(X.shape[1]) # Identity matrix as initial Hessian approximation
        Beta = np.zeros(X.shape[1]).reshape(-1,1) 
        g_prev = l_grad(X,y,Beta).reshape(-1,1)
        # ======= Step 1 - Estimation of the parameters using BFGS optimization =======
        it = 0
        while np.sqrt(g_prev.T @ g_prev) > tol or it < max_iter:
            # compute a proposal for Beta using the BFGS update
            ## find the optimal step size alpha using line search (e.g. backtracking line search)
            # ======= Step 1.1 - Estimation of the step length alpha =======
            direction = D_prev @ g_prev
            a = backtracking_line_search(logl, X, y, Beta_prev, D_prev, g_prev)
            Beta = Beta_prev + a * direction

            # ======= Step 1.2 - Update the Hessian approximation =======
            # Compute the associated gradient and update the Hessian approximation using the BFGS formula
            g = l_grad(X,y,Beta).reshape(-1,1)
            s = Beta - Beta_prev
            yk = g - g_prev

            ys = float(yk.T @ p)          # y^T s
            if ys <= 1e-12:
                D = D_prev               # skip update if curvature condition is not satisfied
            else:
                Dy  = D_prev @ yk
                yDy = float(yk.T @ Dy)
                D = (D_prev
                    + (1.0 + yDy/ys) * (p @ p.T) / ys
                    - (Dy @ p.T + p @ Dy.T) / ys
                )
            
            # Saving previous steps
            Beta_prev = Beta.copy()
            g_prev = g.copy()
            D_prev = D.copy()
            it+=1


        
            






        pass

    def predict(self, X):
        # Implement the prediction procedure for fractional regression
        pass