"""
Title: Communication Subspace Utility Classes
Author: @nehabinish
Created: 11/10/2024 

Description:
Utility script with different classes for finding a communication subspace between two 
brain regions; source and target. 
"""

# Imports for functionality
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

# Set a random seed for reproducibility
random.seed(48)

# Import custom init module
try:
    import init as In
except ImportError:
    In = None

class RRR:
    """
    Class: RRR

    Description:
    The RRR class provides methods for performing reduced rank regression to find a low-dimensional
    mapping between high-dimensional datasets X (source) and Y (target). It enables the identification 
    of a communication subspace between two regions, which may be used to study information exchange 
    between high-dimensional datasets.

    --- NOTE ---
    Most functions in this class are adapted from the matlab codes for 
    "Cortical areas interact through a communication subspace", 
    Semedo et al. (Neuron, 2019). v0.2b MATLAB codes from Joao Semedo 
    https://github.com/joao-semedo/communication-subspace/

    Functions:
        __init__: Initializes the RRR class with datasets X and Y, and optional training/testing sets 
          for cross-validation.
        ReducedRankRegress: Implements the reduced rank regression model, returning a coefficient matrix 
          for projecting X to Y, dimensionality reduction eigenvectors, and initialization options.
        normalized_squared_error: Computes the normalized squared error (NSE) between actual target data 
          and predictions, useful for model evaluation.
        mean_squared_error: Computes the mean squared error (MSE) between actual target data and predictions, 
          providing an alternative metric for model assessment.
        RegressPredict: Computes predicted variables using regression.
        get_ridge_regress: Alternative to python sklearn RidgeRegression functions (not used).
        RegressFitandPredict: Fits a regression model to training data and predicts outcomes on test data.  
        model_select: Selects the optimal regression parameter (alpha) based on cross-validated loss.
    """     
    
    # --------------------------------------------
    # Initialization
    # --------------------------------------------
    def __init__(self, **kwargs):
        """
        Initializes the RRR class with input data and optional cross-validation splits.

        Args:
            X (np.ndarray): Source data for RRR. A 2D array where rows are samples and columns are features.
            Y (np.ndarray): Target data for RRR. A 2D array where rows are samples and columns are features.
            **kwargs: Optional keyword arguments for cross-validation and other parameters.
                crossval (bool): If True, indicates that cross-validation is being performed.
                train (np.ndarray): Training dataset for cross-validation. Required if 'crossval' is True.
                test (np.ndarray): Test dataset for cross-validation. Required if 'crossval' is True.
                subject_alpha (list): Pre-calculated ridge regularisation parameter for subject.

        Raises:
            ValueError: If the input data is not in the correct 2D format or if cross-validation is improperly configured.
        """

        # Cross-validation setup
        if 'crossval' in kwargs and kwargs['crossval']:
            if 'Xtrain' not in kwargs or 'Xtest' not in kwargs:
                raise ValueError("When 'crossval' is specified, both 'train' and 'test' datasets for source must be provided.")
            if 'Ytrain' not in kwargs or 'Ytest' not in kwargs:
                raise ValueError("When 'crossval' is specified, both 'train' and 'test' datasets for source must be provided.")
            
            self.Xtrain = np.array(kwargs['Xtrain'])
            self.Xtest = np.array(kwargs['Xtest'])

            self.Ytrain = np.array(kwargs['Ytrain'])
            self.Ytest = np.array(kwargs['Ytest'])

        else:
            # Convert data to numpy arrays
            X = kwargs['X']
            Y = kwargs['Y']
            self.X = np.array(X)
            self.Y = np.array(Y)

            # Validate the shape of the input data
            if self.X.ndim != 2 or self.Y.ndim != 2:
                raise ValueError("Input data must be 2D arrays with shape [samples x features].")

            Xsamples, Xfeatures = self.X.shape
            Ysamples, Yfeatures = self.Y.shape
            
            # Ensure there are more samples than features
            if Xsamples < Xfeatures or Ysamples < Yfeatures:
                raise ValueError("The number of samples must be greater than or equal to the number of features (samples >= features).")
            
        if 'subject_alpha' in kwargs:
            self.use_alpha = True
            self.subject_alpha = kwargs['subject_alpha']    

    # --------------------------------------------
    # Reduced Rank Regression main function
    # --------------------------------------------
    def ReducedRankRegress(self, X, Y, dim, use_ridge_init=True, scale=False, verbose=False):
        """
        Performs Reduced Rank Regression (RRR) to find a low-dimensional mapping between
        a high-dimensional source (X) and a target (Y) dataset. The goal is to project
        X onto a lower-dimensional subspace that maximally correlates with Y.

        Args:
            X (np.ndarray): Source data matrix (N x p).
            Y (np.ndarray): Target data matrix (N x K).
            dim (list): Vector containing the numbers of predictive dimensions to be tested.
            use_ridge_init (bool): Whether to use Ridge Regression for initialization. Defaults to True.
            scale (bool): Whether to z-score the input data. Defaults to False.
            verbose (bool): Whether to print progress information. Defaults to False.

        Returns:
            B (np.ndarray)    : Mean-adjusted coefficient matrix for the reduced-rank projection 
                                from X to Y, of shape [channels_X, dim].
            B_ (np.ndarray)   : The raw reduced rank coefficient matrix before mean adjustment or 
                                Predictive dimensions ordered by explained variance.
            V (np.ndarray)    : Variance matrix or PCA eigenvectors from the dimensionality 
                                reduction step.

        """

        # Exclude neurons with zero variance - these neurons have the same value for all samples,
        # which can lead to problems in certain mathematical operations, including matrix inversion and linear regression.
        # Specifically, when all the values in a column are the same, the matrix becomes singular,
        # which means it doesn't have an inverse.

        m = np.mean(X, axis=0)
        s = np.std(X, axis=0)  

        non_zero_var_idx = np.where(np.abs(s) >= np.sqrt(np.finfo(s.dtype).eps))[0]

        if len(non_zero_var_idx) != X.shape[1]:
            X = X[:, non_zero_var_idx]
            m = m[non_zero_var_idx]

        n, K = Y.shape  # n is the number of data points | K target dimensionality
        p = X.shape[1]  # p is the source dimensionality

        # subtracting means from all source observations
        M = np.tile(m, (n, 1))                      # number of data points should be same between 
                                                    # the source and target matrix

        # Mean Subtracted matrix
        Z = X - M  

        if verbose:
            print('Size of Mean-subtracted Matrix - {}'.format(np.shape(Z)))

        # for regularisation of ridge regression switch to True
        if use_ridge_init:
            if verbose:
                print('\nUsing Ridge Regression ...')
            # Initialize a range of lambda values for ridge regression
            if self.use_alpha:
                lambda_range = [self.subject_alpha]
            else:    
                lambda_range = np.linspace(0.05, 1000, 1000)

            if verbose:
                print('Using alpha - {}'.format(lambda_range))    

            # number of cross-validation folds
            cvNumFolds = 10

            # Initialise kfold model
            kf = KFold(n_splits=cvNumFolds, random_state=48, shuffle=True)

            # Perform ridge regression with cross-validation
            ridge_init = RidgeCV(alphas=lambda_range, cv=kf, fit_intercept=False)

            # Fit regression model
            if verbose:
                print('Fitting X and Y with shapes - {} and {}'.format(np.shape(X), np.shape(Y)))
            ridge_init.fit(X, Y)
            
            # Retrieve the best alpha (regularization parameter)
            best_alpha = ridge_init.alpha_
            if verbose:
                print(f"Best alpha for this fold RidgeCV: {best_alpha}")

            # Get coefficient matrix or weights
            Bfull = ridge_init.coef_.T              # Exclude intercept term
            if verbose:
                print(np.shape(Bfull))
        else:
            # Perform linear regression Bfull = Z\Y    
            Bfull = np.dot(np.linalg.pinv(Z), Y)    # Bfull is the same as B[OLS], weights of linear regression

        # Perform PCA on the predicted target data
        Yhat = Z @ Bfull
        pca = PCA(whiten=False, svd_solver='auto')
        pca.fit(Yhat)
        V = pca.components_.T

        # Calculate predictive dimensions
        B_ = Bfull @ V

        # Reconstruct mapping matrix B for the requested dimensions
        if dim[0] == 0:
            B = np.zeros((p, K))
        else:
            B = Bfull @ V[:, :dim[0]] @ V[:, :dim[0]].T

        # For multiple tested dimensions
        num_dims = len(dim)
        if num_dims > 1:
            B = np.concatenate([B] + [Bfull @ V[:, :d] @ V[:, :d].T for d in dim[1:]], axis=1)    

        # Mean adjust B
        mean_new = np.array([sum(col) / len(col) for col in zip(*X)])
        mB = np.dot(mean_new, B)

        # Calculate the mean of Y along the first axis
        Y_mean = np.mean(Y, axis=0)

        # Repeat the mean array to match the shape of mB
        Y_mean_repeated = np.tile(Y_mean, (1, num_dims))

        # Calculate the mean-adjusted B
        B_mean_adjusted = Y_mean_repeated - mB

        if verbose:
            print(np.shape(B_mean_adjusted))

        # Stack the mean-adjusted B on top of the original B
        B = np.vstack((B_mean_adjusted, B))
        if verbose:
            print('B after stacking - ' + str(np.shape(B)))

        # If any features were excluded, adjust B_ to the correct size
        if len(non_zero_var_idx) != p:
            full_B_ = np.zeros((X.shape[1], K))
            full_B_[non_zero_var_idx, :] = B_
            B_ = full_B_    

        return B, B_, V, Bfull, best_alpha
    
    # --------------------------------------------
    # Get Normalized Squared Error
    # --------------------------------------------
    @staticmethod
    def normalized_squared_error(Ytest, Yhat):
        """
        Compute the normalized squared error (NSE) between the actual test data (Ytest) and predictions (Yhat).
        The normalized squared error is calculated as the ratio of the residual sum of squares (RSS) to the total 
        sum of squares (TSS), providing a measure of how well the predictions (Yhat) match the actual data (Ytest).
        
        Args:
            Ytest (np.ndarray): Actual test data matrix of shape (N, K), where N is the number of samples
                                and K is the number of target dimensions.
            Yhat (np.ndarray): Predicted data matrix of shape (N, K * numModels), where numModels represents
                                the number of different models being evaluated.
            
        Returns:
            nse (np.ndarray): Normalized squared error for each model, shape (numModels,).
                                Lower values indicate better model performance.
        """
        
        # Get the dimensionality of the target (K) and the number of models (numModels)
        K = Ytest.shape[1]
        numModels = Yhat.shape[1] // K  # Number of models predicted (Yhat is flattened across models)
        
        # Repeat the actual test data (Ytest) for comparison with each model's predictions
        Ytest_repeated = np.tile(Ytest, (1, numModels))  # Shape: (N, K * numModels)
        
        # Compute the squared differences between Ytest and Yhat
        squared_diff = np.square(np.subtract(Ytest_repeated, Yhat))  # Shape: (N, K * numModels)
        
        # Sum the squared differences along the rows (samples) to get the sum of squares per feature
        sum_squared_diff = np.sum(squared_diff, axis=0)  # Shape: (K * numModels,)
        
        # Reshape the sum of squares to separate models (each column corresponds to a model)
        reshaped_sum_squares = sum_squared_diff.reshape(K, numModels, order='F')  # Shape: (K, numModels)
        
        # Sum across the target dimensions (K) to get the residual sum of squares (RSS) for each model
        rss = np.sum(reshaped_sum_squares, axis=0)  # Shape: (numModels,)
        
        # Compute the total sum of squares (TSS)
        N = Ytest.shape[0]
        
        # Calculate the mean of Ytest across samples (mean of each target dimension)
        Ytest_mean = np.mean(Ytest, axis=0)  # Shape: (K,)
        
        # Repeat the mean for each sample to match Ytest's shape
        Ytest_mean_repeated = np.tile(Ytest_mean, (N, 1))  # Shape: (N, K)
        
        # Compute the total sum of squares (TSS) by summing the squared deviations from the mean
        tss = np.sum(np.square(np.subtract(Ytest, Ytest_mean_repeated)))  # Scalar value (sum of all squared deviations)
        
        # Compute the normalized squared error (NSE) for each model
        nse = rss / tss  # Shape: (numModels,)
        
        return nse

    # --------------------------------------------
    # Compute Mean Squared Error
    # --------------------------------------------
    @staticmethod
    def mean_squared_error(Ytest, Yhat):
        """
        Compute the mean squared error (MSE) between the actual test data (Ytest) and predictions (Yhat).  
        The mean squared error is calculated as the average of the squared differences between the actual
        and predicted values, providing a measure of how well the predictions (Yhat) match the test data (Ytest).
        
        Args:
            Ytest (np.ndarray): Actual test data matrix of shape (N, K), where N is the number of samples
                                and K is the number of target dimensions.
            Yhat (np.ndarray): Predicted data matrix of shape (N, K * numModels), where numModels represents
                                the number of different models being evaluated.
        
        Returns:
            mse (np.ndarray): Mean squared error for each model, shape (numModels,).
                                Lower values indicate better model performance.
        """
        
        # Extract the number of samples (N) and the number of target dimensions (K)
        n, K = Ytest.shape
        
        # Calculate the number of models being evaluated
        numModels = Yhat.shape[1] // K  # Assumes Yhat is flattened across models
        
        # Repeat the actual test data (Ytest) for comparison with each model's predictions
        Ytest_repeated = np.tile(Ytest, (1, numModels))  # Shape: (N, K * numModels)
        
        # Compute the squared differences between Ytest and Yhat
        squared_diff = np.square(np.subtract(Ytest_repeated, Yhat))  # Shape: (N, K * numModels)
        
        # Reshape the squared differences to separate models and target dimensions
        squared_diff_reshaped = squared_diff.reshape(n, K, numModels)  # Shape: (N, K, numModels)
        
        # Compute the sum of squared differences across samples (N) and target dimensions (K)
        mse = np.sum(np.sum(squared_diff_reshaped, axis=0), axis=0) / (n * K)  # Shape: (numModels,)
        
        return mse

    # --------------------------------------------
    # Predict using regression
    # --------------------------------------------
    def RegressPredict(self, Y, X, B, noskip=True, verbose=False, **kwargs):
        """
        Computes the predicted target values (Yhat) using the learned regression coefficients (B)
        and calculates the loss between the actual target values (Y) and predicted values (Yhat).
        
        The function supports different loss measures, such as Mean Squared Error (MSE) and
        Normalized Squared Error (NSE). It also adjusts the input data matrix (X) to account for
        an intercept (constant term) in B.

        Args:
            Y (np.ndarray): Target data matrix of shape (N, K).
            X (np.ndarray): Source data matrix of shape (N, p), where N is the number of data points
                            and p is the number of features.
            B (np.ndarray): Coefficient matrix of shape (p + 1, K), where p + 1 accounts for the intercept term
                            and K is the number of target dimensions.
            noskip (bool, optional): If True, uses the specified loss measure in kwargs; otherwise, defaults to NSE.
                                    Defaults to True.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.
            kwargs (dict): Additional keyword arguments, including:
                'LOSSMEASURE' (str): The loss function to use ('MSE' or 'NSE').
        
        Returns:
            loss (float): The computed loss based on the chosen measure.
            Yhat (np.ndarray): The predicted target values, of shape (N, K).
        
        Raises:
            ValueError: If an invalid loss measure is specified in the kwargs.
        """
        
        # Default loss measure
        loss_measure = 'NSE'

        # Since B includes an intercept, we augment X by adding a column of ones or add an intercept term to X
        X_stacked = np.hstack((np.ones((X.shape[0], 1)), X))  # Shape: (N, p + 1)
        
        # Compute predicted target values (Yhat) from X using the learned weights (B)
        Yhat = np.dot(X_stacked, B)  # Shape: (N, K)
        if verbose:
            print(f"Shape of Yhat: {np.shape(Yhat)}")
        
        # Compute loss based on the specified or default loss measure
        if noskip:
            # If 'LOSSMEASURE' is provided in kwargs, use it to determine the loss function
            if 'LOSSMEASURE' in kwargs.keys():
                loss_measure = kwargs['LOSSMEASURE']
            
            # Compute the loss based on the selected measure
            if loss_measure == 'MSE':
                loss = self.mean_squared_error(Y, Yhat)
            elif loss_measure == 'NSE':
                loss = self.normalized_squared_error(Y, Yhat)
            else:
                raise ValueError(f"Invalid loss measure specified: {loss_measure}")
        else:
            # Default to NSE if noskip is False
            loss = self.normalized_squared_error(Y, Yhat)
        
        return loss, Yhat
    
    # --------------------------------------------
    # Get Ridge Regression
    # --------------------------------------------
    @staticmethod
    def get_ridge_regress(Y, X, lambda_, scale=False):
        """
        Fits a Ridge Regression model with regularization parameter lambda to target variables Y 
        and source variables X, returning the mapping matrix B (which includes an intercept).
        
        Args:
            Y (np.ndarray): Target data matrix of shape (N, K).
            X (np.ndarray): Source data matrix of shape (N, p).
            lambda_ (np.ndarray): Vector containing the regularization parameters to be tested (1 x numPar).
            scale (bool): Whether to use variance scaling (z-scoring). Defaults to False.
        
        Returns:
            B (np.ndarray): Extended mapping matrix of shape (p + 1, K * numPar).
        """
        
        # Get dimensions
        n, K = Y.shape
        p = X.shape[1]

        # Calculate mean and standard deviation
        m = np.mean(X, axis=0)
        s = np.std(X, axis=0, ddof=0)
        
        # Handle features with near-zero standard deviation
        idxs = np.where(np.abs(s) < np.sqrt(np.finfo(s.dtype).eps))[0]
        if idxs.size > 0:
            s[idxs] = 1  # Avoid division by zero

        # Expand mean and std for scaling
        M = np.tile(m, (n, 1))
        S = np.tile(s, (n, 1))

        # Scale the data if required
        if scale:
            Z = (X - M) / S
        else:
            Z = X - M

        # Handle features with near-zero standard deviation in Z
        if idxs.size > 0:
            Z[:, idxs] = 1  # Replace with ones

        # Prepare augmented matrices for Ridge regression
        Z_plus = np.vstack((Z, np.sqrt(lambda_[0]) * np.eye(p)))
        Y_plus = np.vstack((Y, np.zeros((p, K))))

        # Calculate the coefficients for the first lambda
        B = np.linalg.solve(Z_plus, Y_plus)

        num_lambdas = len(lambda_)
        if num_lambdas > 1:
            B[-1, K * (num_lambdas - 1)] = 0  # Set the last element of the last column to 0
            for i in range(1, num_lambdas):
                Z_plus[-p:, :] = np.sqrt(lambda_[i]) * np.eye(p)
                B[:, K * i:K * (i + 1)] = np.linalg.solve(Z_plus, Y_plus)

        # Scale back if needed
        if scale:
            B /= s[:, np.newaxis]  # Broadcasting for scaling

        B = np.vstack((np.tile(np.mean(Y, axis=0), (num_lambdas, 1)).T - np.dot(m, B), B))
        
        return B
    
    # --------------------------------------------
    # Regress Fit and Predict
    # --------------------------------------------
    def RegressFitAndPredict(self, regressFun, ndim, verbose=False, **kwargs):

        """
        Fits a regression model to training data and predicts outcomes on test data.    
        This function uses a regression function (e.g., Reduced Rank Regression) to fit a model
        to the training data (Xtrain and Ytrain) and then predicts the target data (Ytest) using
        the test input data (Xtest). It also computes the loss incurred by these predictions,
        allowing comparison of models with different regularization parameters (alpha).
        
        Args:
            regressFun (str): The name of the regression function to use, e.g., 'RRR' for Reduced Rank Regression.
            Xtrain (np.ndarray): Training source data matrix of shape (Ntrain, p), where Ntrain is the number
                                of training samples and p is the number of features (e.g., neural channels).
            Ytrain (np.ndarray): Training target data matrix of shape (Ntrain, K), where K is the target dimensionality.
            Xtest (np.ndarray): Testing source data matrix of shape (Ntest, p), where Ntest is the number of testing samples.
            Ytest (np.ndarray): Testing target data matrix of shape (Ntest, K).
            verbose (bool, optional): If True, enables verbose output. Defaults to False.
            kwargs (dict): Additional keyword arguments to be passed to the regression function or the prediction function.
                            These may include parameters like tolerance, maximum iterations, etc.
            
        Returns:
            loss (np.ndarray): Loss values computed between predicted Ytest and actual Ytest, with shape (1, numPar),
                                where numPar is the number of regularization parameters tested.
            B (np.ndarray): The learned coefficient matrix from the regression model.
        
        --- NOTE ---
        Currently, only Reduced Rank Regression ('RRR') is supported. Extendable to other regression methods.
        """

        Xtrain, Xtest, Ytrain, Ytest = self.Xtrain, self.Xtest, self.Ytrain, self.Ytest

        # Extract additional keyword arguments for the regression or prediction functions
        kwargs_dict = {}
        for kw in sorted(kwargs.keys()):
            kwargs_dict[kw] = kwargs[kw]

        # Check for valid regression function handle
        if regressFun == 'RRR':
            if verbose:
                print('Applying Reduced Rank Regression to training data')
            
            # Fit Reduced Rank Regression model to the training data
            B, B_, V, Bfull, best_alpha = self.ReducedRankRegress(Xtrain, Ytrain, ndim)
            
            # Predict the target data for the test set and compute loss
            if verbose:
                print('Computing loss on test data')
            loss, Y_transformed = self.RegressPredict(Ytest, Xtest, B, verbose=verbose, **kwargs_dict)

        elif regressFun=='ridge':
            
            if verbose:
                print(kwargs['lambda_chosen'])
            B =  self.get_ridge_regress(Ytrain, Xtrain, kwargs['lambdas_chosen'])
            loss = self.RegressPredict(Ytest, Xtest, B, verbose=verbose, **kwargs_dict)
            best_alpha=0
        
        else:
            if verbose:
                print(f"Error: {regressFun} is not a valid regression function handle.")
            loss, B = None, None
        
        if verbose:
            print('Regress Fit and Predict completed!')
        return loss, B, best_alpha
    
    # --------------------------------------------
    # Model Selection
    # --------------------------------------------
    @staticmethod
    def model_select(cv_loss, alpha, verbose=False):
        """
        Selects the optimal regression parameter (alpha) based on cross-validated loss, following the "one-standard-error" rule. 
        The "one-standard-error" rule selects the simplest model (smallest alpha) for which the test performance
        is within one standard error of the best-performing model.

        Args:
            cv_loss (np.ndarray): A 2D array of shape (2, num_par) containing cross-validated loss data. 
                                    The first row represents the mean loss for each regression parameter, 
                                    and the second row represents the standard error of the mean loss.
            alpha (np.ndarray): Model parameters associated with each column in cvLoss. The
                                model parameters in alpha must be ordered in terms of model complexity,
                                i.e., from the simplest to the most complex model. For example, for
                                Ridge regression, alpha is the regularization parameter lambda, which
                                should be in decreasing order. For reduced rank regression, alpha is the
                                number of predictive dimensions, which should be in increasing order.
                                (1 x numPar), numPar is the number of regression parameters used or dimensions.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.
                            
        Returns:
            alpha_opt (float): The optimal regression parameter, selected as the simplest model within 
                                one standard error of the best-performing model.
            opt_loss (float): The average test loss corresponding to the optimal parameter `alpha_opt`.
        """
        
        # Extract the mean cross-validated loss (first row) and find the minimum loss
        loss = cv_loss[0]  # Mean cross-validated loss for each parameter
        min_loss = np.min(loss)  # Minimum loss across all parameters
        min_idx = np.argmin(loss)  # Index of the minimum loss
        
        # Get the standard error of the loss at the minimum loss
        std_error_min_loss = cv_loss[1, min_idx]  # Standard error associated with the minimum loss
        
        # Apply the "one-standard-error" rule to select the simplest model within one std. error of the best model
        # Find the index of the smallest alpha for which the loss is within one std. error of the minimum loss
        alpha_opt_idx = np.where(loss <= min_loss + std_error_min_loss)[0][0]
        
        # Get the optimal alpha (simplest model within one std. error)
        alpha_opt = alpha[alpha_opt_idx]
        
        # Retrieve the corresponding loss for the optimal alpha
        opt_loss = loss[alpha_opt_idx]

        # Print parameters
        if verbose:
            print('Minimum loss across all dimensions - ', min_loss)
            print('Dimension with the minimum loss - ', min_idx)
            print('Std Error for minimum loss - ', std_error_min_loss)
            print('Optimal Dimension', alpha_opt)
            print('Optimal Loss for that dimension', opt_loss)

        return alpha_opt, opt_loss

