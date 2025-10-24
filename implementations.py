import numpy as np

# Function 1: Mean Squared Error using Gradient Descent
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    
    """
    Linear regression using gradient descent (MSE loss).
    Args:
        y (numpy.ndarray): Training labels of shape (N,)
        tx (numpy.ndarray): Training data matrix of shape (N, D)
        initial_w (numpy.ndarray): Initial weights of shape (D,)
        max_iters (int): Number of iterations (gradient descent steps)
        gamma (float): Learning rate
    Returns:
        tuple: (w, loss) where w is the final weight vector (shape (D,)) 
               and loss is the MSE cost at this w.
    """
    w = initial_w.copy()
    N = y.shape[0]

    # Perform gradient descent
    for iter in range(max_iters):
        # Compute the error (residuals)
        error = y - tx.dot(w)

        # Compute the gradient
        grad = -tx.T.dot(error) / N

        # Update weights by moving in the negative gradient direction
        w = w - gamma * grad

    error = y - tx.dot(w)
    loss = np.dot(error, error) / (2 * N)
    return w, loss


# Function 2: Mean Squared Error using Stochastic Gradient Descent
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent (mini-batch size 1).
    Args:
        y (numpy.ndarray): Training labels of shape (N,)
        tx (numpy.ndarray): Training data matrix of shape (N, D)
        initial_w (numpy.ndarray): Initial weights of shape (D,)
        max_iters (int): Number of stochastic updates to perform
        gamma (float): Learning rate
    Returns:
        tuple: (w, loss) where w is the final weight vector and loss is the MSE cost at this w.
    """
    w = initial_w.copy()
    N = y.shape[0]

    for iter in range(max_iters):
        # Randomly pick an index for the sample (mini-batch of size 1)
        i = np.random.randint(0, N)
        # Compute error for this sample
        error_i = y[i] - np.dot(tx[i], w)
        # Compute gradient using this single sample
        grad_i = -error_i * tx[i]
        # Update weights
        w = w - gamma * grad_i
    # Compute final loss (MSE) on the entire dataset
    error = y - tx.dot(w)
    loss = np.dot(error, error) / (2 * N)
    return w, loss

# Function 3: Least Squares
def least_squares(y, tx):
    """
    Least squares regression using normal equations.
    Args:
        y (numpy.ndarray): Training labels of shape (N,)
        tx (numpy.ndarray): Training data matrix of shape (N, D)
    Returns:
        tuple: (w, loss) where w is the optimal weight vector and loss is the MSE cost at this w.
    """
    # Compute the normal equation solution: (X^T X) w = X^T y
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    # Solve for w 
    w = np.linalg.solve(a, b)
    # Calculate the MSE loss for this optimal w
    error = y - tx.dot(w)
    loss = np.dot(error, error) / (2 * y.shape[0])
    return w, loss

# Function 4: Ridge Regression
def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations.
    Args:
        y (numpy.ndarray): Training labels of shape (N,)
        tx (numpy.ndarray): Training data matrix of shape (N, D)
        lambda_ (float): Regularization parameter (strength of L2 penalty)
    Returns:
        tuple: (w, loss) where w is the ridge regression weight vector and loss is the MSE cost at this w (excluding regularization).
    """
    # N, D = tx.shape

    # Compute normal equation with L2 regularization: (X^T X + lambda * N * I) w = X^T y
    # a = tx.T.dot(tx) + lambda_ * N * np.eye(D)
    # b = tx.T.dot(y)
    # w = np.linalg.solve(a, b)
    N, D = tx.shape
    I = np.eye(D)
    A = tx.T @ tx + 2 * lambda_ * N * I
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    # Compute MSE loss on training data (without regularization term)
    error = y - tx.dot(w)
    loss = np.dot(error, error) / (2 * N)
    return w, loss

# Function 5: Logistic Regression
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Binary logistic regression using gradient descent.
    Args:
        y (numpy.ndarray): Training labels of shape (N,) (values 0 or 1)
        tx (numpy.ndarray): Training data matrix of shape (N, D)
        initial_w (numpy.ndarray): Initial weights of shape (D,)
        max_iters (int): Number of iterations (gradient descent steps)
        gamma (float): Learning rate
    Returns:
        tuple: (w, loss) where w is the final weight vector and loss is the logistic loss (negative log-likelihood) at this w.
    """
    # Sigmoid (logistic) function
    # sigmoid = lambda t: 1 / (1 + np.exp(-t))
    from helpersmodels import sigmoid

    w = initial_w.copy()
    N = y.shape[0]

    for iter in range(max_iters):
        # Compute the predicted probabilities
        pred = sigmoid(tx.dot(w)) 

        # Gradient of negative log-likelihood: (1/N) * X^T (pred - y)
        
        grad = tx.T.dot(pred - y) / N

        # Update weights
        w = w - gamma * grad

    # Final loss computation with the updated weights
    loss = -np.mean(
        y * np.log(sigmoid(tx.dot(w))) + (1 - y) * np.log(1 - sigmoid(tx.dot(w)))
    )

    return w, loss

# Function 6: Regularized Logistic Regression
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized (L2) logistic regression using gradient descent.
    Args:
        y (numpy.ndarray): Training labels of shape (N,) (values 0 or 1)
        tx (numpy.ndarray): Training data matrix of shape (N, D)
        lambda_ (float): Regularization strength (L2 penalty coefficient)
        initial_w (numpy.ndarray): Initial weights of shape (D,)
        max_iters (int): Number of iterations (gradient descent steps)
        gamma (float): Learning rate
    Returns:
        tuple: (w, loss) where w is the final weight vector and loss is the logistic loss at this w (excluding regularization).
    """
    sigmoid = lambda t: 1 / (1 + np.exp(-t))
    
    w = initial_w.copy()
    N = y.shape[0]

    for iter in range(max_iters):
        # Compute predictions
        pred = sigmoid(tx.dot(w))

        # Gradient of regularized loss: (1/N)*X^T(pred - y) + lambda * w
        grad = (tx.T.dot(pred - y) / N) + 2 * lambda_ * w

        # Update weights
        w = w - gamma * grad

    # Final loss calculation
    final_pred = sigmoid(tx.dot(w))
    loss = -np.mean(y * np.log(final_pred) + (1 - y) * np.log(1 - final_pred))

    return w, loss
