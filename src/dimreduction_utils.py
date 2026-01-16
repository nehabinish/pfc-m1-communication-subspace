"""
Title: Dimensionality reduction utilities
Author: @nehabinish
Created: 02/10/2024

Description:
Utility functions and class for performing different dimensionality reduction techniques
(example: Factor Analysis (FA) on neural datasets).
"""

# System imports
import numpy as np
import multiprocessing
from numpy.linalg import inv, LinAlgError
from statistics import geometric_mean
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

# Set reproducible seed
rng = np.random.default_rng(1042)

# Import custom init module
try:
    import init as In
except ImportError:
    In = None

class FactorAnalysis:
    """
    Factor Analysis (FA) with EM algorithm for neural data.
    Includes Probabilistic PCA (PPCA) option, cross-validation, and model selection.
    
    --- NOTE ---
    Most functions in this class are adapted from the matlab codes for 
    "Cortical areas interact through a communication subspace", 
    Semedo et al. (Neuron, 2019). v0.2b MATLAB codes from Joao Semedo - fa_util
    https://github.com/joao-semedo/communication-subspace/tree/master/fa_util
    MATLAB code for factor analysis has been made publicly available by Byron Yu 
    and can be downloaded from https://users.ece.cmu.edu/~byronyu/software.shtml.
    """

    def __init__(self, X, q, **kwargs):
        """
        Initialize FactorAnalysis with data and latent dimensions.
        
        Args:
            X (np.ndarray): Data matrix [samples x features]
            q (np.ndarray): Latent dimensionalities to test
            crossval (bool): If True, uses train/test sets for CV
            train (np.ndarray): Training set if crossval is True
            test (np.ndarray): Test set if crossval is True    
        """

        # Latent Dimensions to test
        self.q = q

        # Convert data to numpy array
        self.X = np.array(X)

        # Validate the shape of the input data
        if self.X.ndim != 2:
            raise ValueError("Input data X must be a 2D array with shape [samples x features].")

        nsamples, nfeatures = self.X.shape
        
        # Ensure there are more samples than features
        if nsamples < nfeatures:
            raise ValueError("Error: Number of samples must be greater than or equal to the number of features (nsamples >= nfeatures).")

        # Check if cross-validation is to be performed on the dataset
        if 'crossval' in kwargs and kwargs['crossval']:
            if 'train' not in kwargs or 'test' not in kwargs:
                raise ValueError("When 'crossval' is specified, both 'train' and 'test' datasets must be provided.")
            self.Xtrain = np.array(kwargs['train'])
            self.Xtest = np.array(kwargs['test'])
        else:
            self.Xtrain = self.X              # Default: Use the entire dataset for training
            self.Xtest = None                 # No separate test set if cross-validation is not specified

        self.rng = rng                        # Random number generator for reproducibility    

        return
    
    # ----------------------------
    # Core EM Factor Analysis
    # ----------------------------
    @staticmethod
    def factor_analysis(S, q, method='FA', tol=1e-8, max_iter=1e8):
        """
        This function estimates a latent factor model for a dataset by performing Factor Analysis (FA) 
        or Probabilistic PCA (PPCA) on the sample covariance matrix, using the EM algorithm to iteratively 
        estimate the factor loadings and specific variance.

        Args:
            S (np.ndarray): Covariance matrix of the data (p x p), where p is the number of variables.
            q (int): Number of latent factors to estimate.
            method (str): 'FA' (Factor Analysis) or 'PPCA' (Probabilistic PCA) method. Defaults to 'FA'.
            tol (float): Convergence tolerance for log-likelihood. Defaults to 1e-8.
            max_iter (float): Maximum number of iterations for the EM algorithm. Defaults to 1e8.

        Returns:
            L (np.ndarray): Estimated factor loadings matrix (p x q).
            psi (np.ndarray): Estimated specific variance (p,).
            log_like (float): Log-likelihood of the model at the final iteration.
        """
        # p is the dimensionality of the data
        p = S.shape[0]

        # C_MIN_FRAC_VAR is the minimum fraction of variance for stability for psi update stability ensurance.
        C_MIN_FRAC_VAR = 0.01 

        # Scale the covariance matrix
        if np.linalg.matrix_rank(S) == p:
            # If the matrix rank is full (rank = p), use Cholesky decomposition to compute the geometric mean of diagonal elements.
            scale = np.exp(2 * np.sum(np.log(np.diag(np.linalg.cholesky(S)))) / p)
        else:
            # Otherwise, rank-deficient case, Eigenvalue decomposition and the geometric mean of the smallest non-zero eigenvalues.
            r = np.linalg.matrix_rank(S)
            eigvals = np.sort(np.linalg.eigvalsh(S))[::-1]  # Sort in descending order
            scale = geometric_mean(eigvals[:r])             # Geometric mean of the top r eigenvalues

        # Initialize factor loadings matrix L (p x q), using a scaled random initialization
        L = rng.standard_normal(size=(p, q)) * np.sqrt(scale / q)

        # psi is the specific variance (uniqueness), initialized as the diagonal of the covariance matrix
        psi = np.diag(S)

        # Variance floor to avoid degenerate solutions for psi (ensures numerical stability)
        var_floor = C_MIN_FRAC_VAR * psi

        # Identity matrix (q x q), used in the EM algorithm's calculations
        I = np.eye(q)

        # Log-likelihood constants (c) for simplifying the log-likelihood computation
        c = -p / 2 * np.log(2 * np.pi)
        log_like = 0

        # EM Algorithm: Iterate over a maximum number of iterations or until convergence
        for i in range(int(max_iter)):
            # E-step: Estimate the latent variables given the current parameters
            invPsi = np.diag(1.0 / psi)
            invPsiTimesL = invPsi @ L
            invC = invPsi - invPsiTimesL @ np.linalg.inv(I + L.T @ invPsiTimesL) @ invPsiTimesL.T

            # Compute V and intermediate variables for M-step
            V = invC @ L
            StimesV = S @ V

            # Expected value of Z^T * Z, where Z is the latent variable
            EZZ = I - V.T @ L + V.T @ StimesV

            # Log-likelihood computation (for convergence check)
            prev_log_like = log_like

            # Cholesky decomposition to calculate the log-determinant term
            chol_invC = np.linalg.cholesky(invC)

            # log-determinant
            ldm = np.sum(np.log(np.diag(chol_invC)))

            # Full log-likelihood expression for Gaussian factor model
            log_like = c + ldm - 0.5 * np.sum(invC * S)

            # Check for convergence based on the log-likelihood improvement
            if i <= 1:
                baseLogLike = log_like
            elif (log_like - baseLogLike) < (1 + tol) * (prev_log_like - baseLogLike):
                break  # Convergence criterion satisfied

            # M-step: Update factor loadings (L) and specific variances (psi)
            L = StimesV @ np.linalg.inv(EZZ)

            # Update psi
            psi = np.diag(S) - np.sum(StimesV * L, axis=1)

            # Enforce a variance floor for psi to avoid near-zero values
            if method.upper() == 'PPCA':
                # For PPCA, psi is forced to be a scalar (shared across all dimensions)
                psi = np.full_like(psi, psi.mean())
            else:
                # For FA, ensure psi remains non-negative and above the variance floor
                psi = np.maximum(psi, var_floor)

        return L, psi, log_like
    
    # --------------------------------------------
    # Log-likelihood
    # --------------------------------------------
    @staticmethod
    def logdet(A):
        """
        This function computes the logarithm of the determinant of a positive-definite matrix 
        using the Cholesky decomposition. This method is numerically more stable and efficient 
        for large matrices compared to directly computing the determinant.

        Args:
            A (np.ndarray): Positive-definite matrix (n x n) whose log determinant is to be computed.

        Returns:
            y (float): The log determinant of the matrix A.
        """
        # Initialize result
        y = 0

        # Perform Cholesky decomposition of matrix A (A = U.T @ U)
        U = np.linalg.cholesky(A)

        # Compute log-determinant
        y = 2 * np.sum(np.log(np.diag(U)))

        return y

    # --------------------------------------------
    # Multivariate normal log-likelihood
    # --------------------------------------------
    def mvn_log_likelihood(self, X, m, S):
        """
        Computes the log-likelihood of the data X under a multivariate Gaussian distribution 
        with mean vector m and covariance matrix S.

        Args:
            X (np.ndarray): Data matrix (N x p) where N is the number of data points and p is the dimensionality.
            m (np.ndarray): Mean vector (1 x p).
            S (np.ndarray): Covariance matrix (p x p).

        Returns:
            log_likelihood (float): Log-likelihood of the data under the multivariate Gaussian distribution.
        """
        # Shape of test/train data
        N, p = X.shape

        # Center the data by subtracting the mean from each row
        M = np.tile(m, (N, 1))          # Expand the mean to match the number of rows in X
        X_centered = X - M              # Centered data

        # Log-likelihood computation
        log_det_S = self.logdet(S)      # Log determinant of covariance matrix
        S_inv = inv(S)                  # Inverse of covariance matrix
        
        # Mahalanobis distance term: X_centered.T @ S_inv @ X_centered for each data point
        mahalanobis_term = np.sum(X_centered @ S_inv * X_centered, axis=1)

        # Compute total log-likelihood
        log_likelihood = -0.5 * (N * p * np.log(2 * np.pi) + N * log_det_S + np.sum(mahalanobis_term))

        return log_likelihood
    
    # --------------------------------------------
    #  FA log-likelihood on test data
    # --------------------------------------------
    def factor_analysis_test_log_likelihood(self, Xtrain, Xtest, method='FA'):
        """
        Applies Factor Analysis models with latent dimensionalities q to Xtrain
        and computes the log-likelihood of the data in Xtest under these models.

        Args:
            Xtrain (np.ndarray): Training data matrix (Ntrain x p).
            Xtest (np.ndarray): Test data matrix (Ntest x p).
            method (str): 'FA' for Factor Analysis, 'PPCA' for Probabilistic PCA.

        Returns:
            logLike (np.ndarray): Array of log-likelihood values corresponding to each latent dimensionality q 
                                (1 x numDims). NaN is returned if the Cholesky decomposition fails or if 
                                certain conditions are met.
        """

        # Latent dimensions to test
        q = self.q

        # Mean of training data
        m = np.mean(Xtrain, axis=0)     

        # Covariance matrix of training data              
        S = np.cov(Xtrain, rowvar=False, bias=True)

        # Number of latent dimensions to evaluate
        numDims = len(q)

        # Initialise log-likelihood array
        logLike = np.zeros(numDims)

        for i in range(numDims):
            if q[i] == 0:
                # Diagonal covariance matrix for zero latent dimensions
                Psi = np.diag(np.diag(S))
                try:
                    # Cholesky Decomposition
                    np.linalg.cholesky(Psi)
                    # Log-likelihood calculation
                    logLike[i] = self.mvn_log_likelihood(Xtest, m, Psi)
                except LinAlgError:
                    # Handle decomposition failure
                    logLike[i] = np.nan
            else:
                # Factor Analysis for non-zero latent dimensions
                L, psi, _ = self.factor_analysis(S, q[i])

                # Check for small specific variances
                idxs = np.abs(psi) < np.sqrt(np.finfo(psi.dtype).eps)
                if np.any(idxs):
                    # Assign NaN if there are small specific variances
                    logLike[i] = np.nan
                    continue

                # Diagonal covariance matrix from specific variances
                Psi = np.diag(psi)
                # Covariance matrix for the multivariate normal distribution
                C = L @ L.T + Psi

                try:
                    # Try Cholesky Decomposition
                    np.linalg.cholesky(C)
                    logLike[i] = self.mvn_log_likelihood(Xtest, m, C)
                except LinAlgError:
                    # Handle decomposition failure
                    logLike[i] = np.nan

        # Return final log-likelihood values
        return logLike
    
    # --------------------------------------------
    # Cross-validation
    # --------------------------------------------
    def CrossValFa(self, cvNumFolds=10, parallel=True):
        """
        Perform cross-validation for Factor Analysis model.

        Args:
            X (np.ndarray): Data matrix (N x p).
            cvNumFolds (int): Number of folds for cross-validation.
            parallel (bool): Switch to parallel processing.

        Returns:
            cvLoss: Cumulative shared variance explained by the latent dimensions.
            cvLogLike: Cross-validated log-likelihood.
        """

        # Latent dimensions to test
        q = self.q

        # Get Data 
        X = self.X
        q = np.sort(q)
        cvLogLike = np.zeros((cvNumFolds, len(q)))

        # K-Fold Cross-Validation
        kf = KFold(n_splits=cvNumFolds)

        # Parallel processing for cross-validation
        if parallel:
            n_cores = max(1, multiprocessing.cpu_count() - 2)  
            cvLogLike = Parallel(n_jobs=n_cores)(delayed(self.factor_analysis_test_log_likelihood)(
                self.X[train_idx], self.X[test_idx]
            ) for train_idx, test_idx in kf.split(self.X))

            # If using joblib, cvLogLike will be a list; convert to an array
            cvLogLike = np.array(cvLogLike)
        else:
            print('\nSerial Analysis ...')
            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                Xtrain, Xtest = X[train_idx], X[test_idx]
                cvLogLike[fold] = self.factor_analysis_test_log_likelihood(Xtrain, Xtest, q)

        # Calculate shared variance explained
        S = np.cov(X, rowvar=False)
        qMaxIdx = np.nanargmax(np.nanmean(cvLogLike, axis=0))
        qMax = q[qMaxIdx]

        if qMax == 0:
            cvLoss = np.nan
            explained_variance = np.nan
        else:
            L, _, _ = self.factor_analysis(S, qMax)
            d = np.sort(np.linalg.eigvalsh(L @ L.T))[::-1]  # Sort in descending order
            
            cvLoss = (1 - np.cumsum(d) / np.sum(d)).T
            explained_variance = d / np.sum(d)              # Fraction of explained variance for each dimension

            if q[0] == 0:
                cvLoss = np.concatenate((np.array([1]), cvLoss), axis=0)
                cvLoss = cvLoss[q]
            else:
                cvLoss = cvLoss[q-1]

        return cvLoss, cvLogLike, explained_variance
    
    # --------------------------------------------
    # Model Selection
    # --------------------------------------------
    def FactorAnalysisModelSelect(self, cvLoss):

        """
        Selects the optimal latent dimensionality q based on cross-validation loss values.

        Args:
            - cvLoss (list or np.ndarray): Cross-validation loss values (1 x numDims).

        Returns:
            - qOpt (int or float): The optimal latent dimensionality q based on the specified variance threshold.
            - Returns 0 if any value in cvLoss is NaN.
        """

        # Latent dimensions to test 
        q = self.q

        # CrossValidation Loss
        cvLoss = np.array(cvLoss)

        # Variance Threshold
        var_threshold = 0.95

        # Check for NaN values in cvLoss
        if np.any(np.isnan(cvLoss)):
            qOpt = 0
        else:
            # Find the optimal q based on the condition
            qOpt = q[np.argmax(1 - cvLoss > var_threshold)]

        return qOpt