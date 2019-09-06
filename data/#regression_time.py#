import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import time

from sklearn.utils import check_X_y
from sklearn.metrics.pairwise import rbf_kernel,polynomial_kernel,linear_kernel
from sklearn.base import BaseEstimator
import time


def preprocess_data(X ,y, fit_intercept, scale):
    """Center and scale the data.

    Parameters
    ----------
    X : ndarray, dim(n,p)
    Design matrix with example in row and features in column

    y : ndarray, dim(1)
    vector of the observed values, references

    fit_intercept : bool
    if True X and y are centered

    scale:
    if True X and y are scaled by their respective standard deviations

    Returns
    -------
    X_out : ndarray(n,p)
    X processed according to the states the booleans fit_intercept and scale

    y_out : ndarray, dim(1)
    y processed

    X_offset : ndarray, dim(p)
    mean of X according to the examples

    X_scale : ndarray, dim(p)
    standard deviations of the featrures of X according to the examples

    y_offset : float
    mean of y 
 
   """
    X_offset = np.zeros(X.shape[1])
    X_scale = np.ones(X.shape[1])
    y_offset = 0

    if fit_intercept is True:
            X_offset = X.mean(0)
            y_offset = y.mean()
    
    if scale is True:
        X_scale = X.std(0)

    X_out = (X-X_offset)/X_scale
    y_out = y-y_offset
    
    return X_out, y_out, X_offset, X_scale, y_offset
        

class BayesianRidge(BaseEstimator):
    """Bayesian implementation of the linear regression, cf Bishop 2006, Pattern 
    recognition in machine learning.

    The optmization is done from the whole dataset by maximizing the marginal 
    likelihood/evidence. 
    The prior on w is assumed to be gaussian with zeros mean and constant
    precision alpha  :    p(w|alpha) = N(0, alpha^-1*Id)  
    
    Parameters
    ----------
    fit_intercept : bool, defaut(True)
    if True X and y are centered

    scale : bool
    if True X is scaled by its respective standard deviation vector

    Attributes
    ----------
    coef_ : ndarray, dim(p)
    weight vector
    
    gamma_ : float
    effective number of parameters

    beta_ : float
    noise inverse variance
    
    """
    def __init__(self,fit_intercept=True,scale=False):
        self.name_='BayesianRidge'
        self.fit_intercept=fit_intercept
        self.scale=scale

        
    def fit(self,PHI,t):
        """Maximize the evidence of a given model tunned with params.
        
        Parameters
        ----------
        PHI : ndarray, dim(N,M)
        Data matrix, with M variables in column.
        
        t : array, dim(N)
        Target/reference values. 
        
        Returns
        -------
        self

        """
        PHI , y = check_X_y(PHI,t,y_numeric=True)
        N , M = PHI.shape
        
        #Preprocess the data
        PHI, t, self.x_mean_, self.x_std_, self.intercept_=preprocess_data(
            PHI, t, fit_intercept=self.fit_intercept, scale=self.scale)
        
        nitermax=50
        tol=1e-3
        PHItt = PHI.T.dot(t)
        PHItPHI = PHI.T.dot(PHI)

        start = time.time()
        eigval, eigvec = np.linalg.eigh(PHItPHI)
        diff = time.time()-start
        print("DIAG in s :", diff)
        
        
        #Initialize the hyperparameters
        #Begin with an infinitely broad prior
        self.alpha_ = 1e-6 
        self.beta_ = 1/(np.var(t)*0.1)
        
        i=0
        log_marginal_likely_hood = np.empty(nitermax)

        delta_alpha_= 1
        delta_beta_= 1
        crit = 1
        #Maximize the marginal likelihood
        while(crit > tol and i < nitermax):
            
            delta_alpha_  = -self.alpha_
            delta_beta_   = -self.beta_
            #Comptute the posterior statistics
            temp = self.alpha_ * np.eye(M) +  PHItPHI * self.beta_
            
            self.covariance_ = np.linalg.inv(temp)
            self.coef_ =  self.covariance_.dot(PHItt) * self.beta_
            #print(self.coef_)
            #Supress in a fute update
            # self.coef_, self.covariance_ = self.posterior(PHI, t)
            #Update alpha
            eigvali = (eigval*self.beta_).real
            gamma = np.sum( eigvali/ (self.alpha_ + eigvali) )
            print("gamma  = ", gamma)
            self.alpha_ = gamma / self.coef_.T.dot(self.coef_)
            #Update beta
            temp = t-PHI.dot(self.coef_)
            self.beta_ = (N-gamma) / temp.T.dot(temp) 
            delta_alpha_ += self.alpha_
            delta_beta_  += self.beta_
            crit = abs(delta_alpha_/self.alpha_ + delta_beta_/self.beta_)
            i +=1
            print(i)
                        
        print("i=",i)
        self.gamma = gamma
        return self


    def posterior(self,PHI,t):
        """Compute the unnormalized posterior distribution
        
        Parameters
        ----------
        PHI : ndarray, dim(N,M)
        Data matrix, with M variables in column.
        
        t : array, dim(N)
        Target/reference values. 
                
        Returns
        -------
        mn : array, dim(M)
        Mean of the posterior pdf

        Sn : ndarray, dim(M,M)
        Covariance matrix of the posterior pdf
        
        """
        So_inv = (self.alpha_) * np.eye(PHI.shape[1])
        Sn = np.linalg.inv( So_inv +  self.PHItPHI * self.beta_)
        mn =  Sn.dot(self.PHItt) * self.beta_
        
        return mn, Sn


    def log_marginal_likelihood(self,PHI,t):
        """Compute the log marginal likelihood of the model given the current 
        parameters -> alpha, beta
        
        parameters
        ----------
        PHI : ndarray, dim(N,M)
        Data matrix, with M variables in column
        
        t : array, dim(N)
        target/reference values 
        
        returns
        -------
        lml : float
        log mariginal likelihood

        """
        N, M = PHI.shape
        A = self.alpha_ * np.eye(M) + self.beta_ * self.PHItPHI

        Emn = self.beta_* ((t-PHI.dot(self.coef_))**2).sum() + \
              self.alpha_ * self.coef_.T.dot(self.coef_)
        
        lml = M * np.log(self.alpha_)
        lml += N * np.log(self.beta_)
        lml -= Emn
        lml -= np.log(np.linalg.det(A))
        lml -= N * np.log(2*np.pi)
        lml *= 0.5/N
        
        return lml
        
    def predict(self,PHI, return_std=False):
        """Predict according to the predictive distribution after marginalization
        p(t|T,phi,alpha,beta) = N(mn.T*phi, s2)
        with s2 = 1/beta + phi.T * Sn * phi
        
        parameters
        ----------
        phi : array or scalar, dim(N,p)
        input data, must be in row

        returns
        -------
        t : array or scalar
        predicted value

        """
        #Manage the case of a unique point
        if PHI.ndim == 1:
            PHI = PHI[np.newaxis,:]

        PHI = (PHI-self.x_mean_) / self.x_std_
        t = PHI.dot(self.coef_) + self.intercept_

        if not return_std:
            return t

        else:
            s2 = np.diag(1/self.beta_ + PHI.dot(self.covariance_).dot(PHI.T))
            return t, np.sqrt(s2)        

    def score(self,X,y):
        """Compute the RMSE
        
        Parameters
        ----------
        X : ndarray, dim(N,M)
        points to predict in rox
        
        y: array, dim(N)
        labels
        
        Returns
        -------
        rmse : float
        """
        return np.mean((y-self.predict(X))**2)**0.5
    
        
class RVMRegression(BaseEstimator):
    """Relevance Vector Machine for regression, cf Bishop 2006, chapter 7,
    Sparse kernel machine, Pattern recognition in machine learning.

    Parameters
    ----------
    fit_intercept : bool
    If True X and y are centered.

    scale :bool
    If True X is scaled by its respective standard deviation vector.

    alpha_threshold : float
    Value of alpha from which a feature is considered to be equal
    to 0 with certainty.

    kernel : string, default(None)
    Specifies the kernel. Available, rbf, polynomial, linear.
    
    gamma : float, default(None)
    Kernel lenght scale. By default gamma is 1.0/n_features.
    
    degree : int, default(3)
    Degree of the polynomial kernel.

    Attributes
    ----------
    coef_ : ndarray, dim(p)
    Weight vector.
    
    act_set : ndarray, bool
    If True spots an avtive basis function.
    
    alpha_ : ndarray, dim(p)
    Regularization parameters.
    
    beta_ : scalar
    Noise inverse variance.
    """

    def __init__(self,fit_intercept=True,scale=False,alpha_threshold=1e4,
                 kernel=None,
                 gamma=None,
                 degree=3):
        self.alpha_threshold = alpha_threshold
        self.name = 'RVM'
        self.fit_intercept=fit_intercept
        self.scale=scale
        self.kernel=kernel
        self.gamma=gamma
        self.degree=degree

    def fit(self,PHI,t):
        """Maximize the evidence of a given model tunned with params
        
        PHI : array, dim(n,p)
        vector or matrix of the input space after a transformation phi(X).
        
        t : array, dim(n)
        vector of the observed values, references.

        """
        PHI , y = check_X_y(PHI,t,y_numeric=True)

        #PHI becomes the Gram matrix if a kernel a is applied
        if self.kernel is not None:
            self.PHI = PHI #Keep the original PHI as attribute for prediction
            PHI = self.apply_kernel(PHI,PHI)
        
        #Preprocess the data
        PHI, t, self.x_mean_, self.x_std_, self.intercept_=preprocess_data(
            PHI, t, fit_intercept=self.fit_intercept, scale=self.scale)

        N , M = PHI.shape
        nitermax=50
        tol=1e-5
        self.evidence = np.zeros(nitermax)
        eps = np.finfo(float).eps
        i=0
        L=1
        crit=1
        
        # Initialize the hyperparameters
        # Begin with an infinitly broad prior
        self.alpha_ = 1e-6*np.ones(M) 
        self.beta_ = 1/(0.1 * np.var(t))
        
        self.act_set = np.ones(M,dtype=bool)
        while( crit > tol and i < nitermax):
            crit = -L
            self.act_set = self.alpha_ < self.alpha_threshold
            #Comptute the posterior statistics
            mn, Sn = self.posterior(PHI, t)
                
            #Update the alpha i
            gammai = 1-Sn.diagonal()*self.alpha_[self.act_set]
            self.alpha_[self.act_set] = gammai/mn**2
            
            #Update beta
            temp = t-np.dot(PHI[:,self.act_set],mn)
            self.beta_ = (N-gammai.sum()) / temp.T.dot(temp) 

            self.coef_, self.covariance_ = mn, Sn
            L=np.linalg.norm(mn)
            crit = abs(crit+L)/L
            self.evidence[i]=np.linalg.norm(L)
            print("gammai.size = ", gammai.size, "gammai.sum() = ",(gammai.sum()))
            i +=1

        print("i = ",i) # FIX ME
        self.coef_ = np.zeros(M)
        self.coef_[self.act_set] = mn
        self.Sn = Sn
        self.covariance_ = np.zeros(M) #covariance as matrix can change
        self.covariance_[self.act_set] = np.sqrt(Sn.diagonal())
        self.covariance_ = np.diag(self.covariance_)
        self.gammai = gammai
        
        return self


    def posterior(self,PHI,t):
        """Compute the statistics of the posterior distribution

        Parameters
        ----------
        PHI : array, dim(n,p)
        vector or matrix of the input space after a transfo
        
        t : array, dim(n)
        target vector, labels in hot encoding

        Returns
        -------
        Statistics of the posterior distribution
        
        """
        act = self.act_set
        A = np.diag(self.alpha_[act]) 
        Sn = np.linalg.inv( A +  np.dot(PHI[:,act].T,PHI[:,act]) * self.beta_)
        mn =  np.dot( np.dot(Sn,PHI[:,act].T), t) * self.beta_

        return mn, Sn


    def log_marginal_likelihood(self,PHI,t):
        """Compute the log marginal likelihood of the model given the 
        current  parameters -> alpha, beta.
        
        Parameters
        ----------
        PHI : ndarray, dim(N,M)
        Data matrix, with M variables in column.
        
        t : array, dim(N)
        Target/reference values. 
        
        Returns
        -------
        lml : float
        log mariginal likelihood.
        
        """
        act = self.act_set
        N, M = PHI[:,act].shape
        A = np.diag(self.alpha_[act])
        lml  = -np.log(np.linalg.det(self.covariance_))
        lml -= N*np.log(self.beta_)
        lml -= np.log((A.diagonal()).prod())
        lml *= 0.5
        lml += self.beta_*t[:,np.newaxis].T.dot(t-PHI[:,act].dot(self.coef_))

        return lml/N
                
                
    def predict(self,PHI,return_std=False):
        """Predict according to the predictive distribution after 
        marginalization
        with s2 = 1/beta + phi.T * Sn * phi
        
        Parameters
        ----------
        phi : array or scalar, dim(N,M)
        input data, must be in row.

        return_std : boolean, default(False)
        If True return the standard deviation of the prediction.

        Returns
        -------
        t : array or scalar
        Predicted value.
        
        std : array or scalar
        Standard deviation of the predicted  value.
        
        """
        #Manage the case of a unique point
        if PHI.ndim == 1:
            PHI = PHI[np.newaxis,:]

        if self.kernel is not None:
            PHI=self.apply_kernel(self.PHI,PHI)

        PHI=(PHI-self.x_mean_)/self.x_std_
            
        t = PHI[:,self.act_set].dot(self.coef_[self.act_set])+self.intercept_
        if not return_std:
            return t

        # Return the standard deviation of the prediction
        else:
            O = PHI[:,self.act_set].dot(self.Sn).dot(PHI[:,self.act_set].T)
            s2 =  np.diag(1/self.beta_ + O)
            return t, np.sqrt(s2)


    def score(self,X,y):
        """Compute the RMSE
        
        Parameters
        ----------
        X : ndarray, dim(N,M)
        points to predict in rox
        
        y: array, dim(N)
        labels
        
        Returns
        -------
        rmse : float
        """
        return np.mean((y-self.predict(X))**2)**0.5


    def apply_kernel(self,X,Y):
        """Apply kernel function between two data matrix.
        
        Parameters
        ----------
        X1 : ndarray, dim(n1,p)
        Data matrix with points in row.
        
        Y : ndarray, dim(n2,p)
        Data matrix with point in row.
        
        kernel : string
        
        params : dict
        Dictionnary that contains the parameters of the kernel.
        cf sklearn documentation of each kernel
        
        Returns
        -------
        Gram matrix : ndarray, dim(n1,n2).
        
        """
        if self.kernel is 'rbf':
            return rbf_kernel(X,Y,gamma=self.gamma).T
        elif self.kernel is 'polynomial':
            return polynomial_kernel(X,Y,gamma=self.gamma,degree=self.degree).T
        elif self.kernel is 'linear':
            return linear_kernel(X,Y).T
                        





    
    
