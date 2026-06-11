import numpy as np
import pandas as pd
from scipy import sparse
from types import SimpleNamespace

class FracRegressionEstimation:
    """Logistic model object for continuous target variables"""
    def __init__(self):
        self.beta_ = None
        self.se_ = None
        self.var_ = None
        self.feature_names_ = None
        self.curv_matrix_ = None
        self.n_iter_ = 0

    @staticmethod
    def data_prep(X):
        """

        Parameters
        ----------
        X :
            

        Returns
        -------

        """
        if isinstance(X, pd.DataFrame):
            features = X.columns.tolist()
            X = sparse.csr_matrix(np.asmatrix(X))  
        elif isinstance(X, np.ndarray):
            features = [f'X{i}' for i in range(X.shape[1])]
            X = sparse.csr_matrix(np.asmatrix(X))    
        elif sparse.issparse(X):
            features = [f'X{i}' for i in range(X.shape[1])]
            X = X.tocsr() 
        else:      
            raise ValueError("Unsupported data type for X. " \
                "               Please provide a pandas DataFrame or a numpy array.")
        return X, features

    def fit(self, X, y, regularization=None, lambda_=0.1, add_intercept = True):
        """

        Parameters
        ----------
        X :
            
        y :
            
        regularization :
             (Default value = None)
        lambda_ :
             (Default value = 0.1)
        add_intercept :
             (Default value = True)

        Returns
        -------

        """
        # Define Backtracking line search function
        def backtracking_line_search(logl, X, y, Beta_prev, D, g,
                                alpha0=1.0, rho=0.5, c=1e-4, min_alpha=1e-8):
            """

            Parameters
            ----------
            logl :
                
            X :
                
            y :
                
            Beta_prev :
                
            D :
                
            g :
                
            alpha0 :
                 (Default value = 1.0)
            rho :
                 (Default value = 0.5)
            c :
                 (Default value = 1e-4)
            min_alpha :
                 (Default value = 1e-8)

            Returns
            -------

            """
            direction = D @ g                          # ascend direction
            grad_dot_dir = float(g.T @ direction)        # directional derivative (scalar)
            if grad_dot_dir <= 0:
                return 0.0                               # not a ascend direction
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
        X, features = self.data_prep(X)

        # add intercept if True
        if  add_intercept:
           intercept = sparse.csr_matrix(np.ones((X.shape[0],1)))
           X = sparse.hstack([intercept, X]).tocsr()
           features = ["Intercept"] + features 
           
        y = np.asarray(y).reshape(-1, 1)
        if regularization is None:
            logl = lambda X,y,Beta: np.sum(y * (X @ Beta) - np.logaddexp(0,X @ Beta))
            l_grad = lambda X,y,Beta: X.T @ (y - 1/(1+np.exp(-X @ Beta))) # Convex Function
        elif regularization == 'L2':
            logl = lambda X,y,Beta: np.sum(y * (X @ Beta) - np.logaddexp(0,X @ Beta)) - lambda_ * np.sum(Beta**2)
            l_grad = lambda X,y,Beta: X.T @ (y - 1/(1+np.exp(-X @ Beta))) - 2 * lambda_ * Beta # Convex Function
        tol = 1e-5
        max_iter = 1000
        # Starting values for the optimization
        D_prev = np.eye(X.shape[1]) # Identity matrix as initial Hessian approximation
        Beta_prev = np.zeros(X.shape[1]).reshape(-1,1) 
        g_prev = l_grad(X,y,Beta_prev).reshape(-1,1)
        # ======= Step 1 - Estimation of the parameters using BFGS optimization =======
        it = 0
        while np.sqrt(g_prev.T @ g_prev) > tol and it < max_iter:
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
            yk = - (g - g_prev)

            ys = float(yk.T @ s)          # y^T s
            if ys <= 1e-12:
                D = D_prev               # skip update if curvature condition is not satisfied
            else:
                Dy  = D_prev @ yk
                yDy = float(yk.T @ Dy)
                D = (D_prev
                    + (1.0 + yDy/ys) * (s @ s.T) / ys
                    - (Dy @ s.T + s @ Dy.T) / ys
                )
            
            # Saving previous steps
            Beta_prev = Beta.copy()
            g_prev = g.copy()
            D_prev = D.copy()
            it+=1

        # storing final parameters 
        self.beta_ = pd.Series(Beta.copy().flatten(), index = features)
        self.curv_matrix_ = D.copy()
        self.n_iter_ = it
        self.feature_names_ = features
        # ======= Step 1.3 - Compute Standard Errors =======
        # Compute the standard errors under robust sandwich estimator formula
        # Define the scores #
        u = y - 1/(1+np.exp(-X @ Beta)) # scores
        U =  X.T @ X.multiply(u**2)
        # Compute the robust sandwich estimator for standard errors
        self.var_ = pd.DataFrame(self.curv_matrix_ @ U @ self.curv_matrix_
        , index = features)
        self.se_  = pd.Series(
            np.sqrt(np.diag(self.var_.values)), index = features)
        return self

    def predict(self, X):
        """

        Parameters
        ----------
        X :
            

        Returns
        -------

        """
        # Implement the prediction procedure for fractional regression
        if self.beta_ is None:
            raise ValueError("Model is not fitted yet.")
        elif isinstance(X, pd.DataFrame):
            try:
                y = 1 / (1 + np.exp(-X.loc[:, self.beta_.index].values @ self.beta_.values.reshape(-1,1)))
            except KeyError:
                raise ValueError("The input DataFrame must contain the same features as the training data.")

    