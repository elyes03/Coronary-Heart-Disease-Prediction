from implementations import * 

def split_train(X, y, test_size=0.2, random_state=None):
    """
    Split dataset into training and test sets.
    Shuffles the data, uses 'test_size' to define the split ratio,
    and 'random_state' for reproducibility.
    Returns X_train, X_test, y_train, y_test.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    n_test = int(test_size * n_samples)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

def accuracy(y_true, y_pred):
    """
    Compute classification accuracy.
    Compares true and predicted labels and returns the proportion of correct predictions.

    Args:
        y_true (np.ndarray): True labels (0/1 or -1/1).
        y_pred (np.ndarray): Predicted labels (same shape).

    Returns:
        float: Accuracy value between 0 and 1.
    """
    
    assert y_true.shape == y_pred.shape
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total 

def f1_score(y_true, y_pred):
    """
    Calculates the F1-score between true labels and predicted labels.

    Parameters:
    y_true (numpy array): True labels (-1 or 1)
    y_pred (numpy array): Predicted labels (-1 or 1)

    Returns:
    float: F1-score
    """
   
    tp = np.sum((y_true == 1) & (y_pred == 1))
   
    fp = np.sum((y_true == -1) & (y_pred == 1))
   
    fn = np.sum((y_true == 1) & (y_pred == -1))

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    if (precision + recall) == 0:
        return 0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def f1_scorelog(y_true, y_pred):
    """
    Compute the F1-score between true and predicted labels.

    Args:
        y_true (np.ndarray): True labels (0 or 1).
        y_pred (np.ndarray): Predicted labels (0 or 1).

    Returns:
        float: F1-score value between 0 and 1.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    if (precision + recall) == 0:
        return 0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def optimize_threshold(y_true, y_scores):
    """
    Find the optimal decision threshold that maximizes the F1-score.

    Iterates over 100 evenly spaced thresholds between the min and max of y_scores,
    evaluates F1-score and accuracy at each, and returns the best threshold.

    Args:
        y_true (np.ndarray): True labels (-1 or 1).
        y_scores (np.ndarray): Continuous prediction scores.

    Returns:
        tuple: (best_threshold, best_f1, best_accuracy)
    """
    best_threshold = None
    best_f1 = 0
    best_accuracy = 0

    thresholds = np.linspace(np.min(y_scores), np.max(y_scores), 100)

    for threshold in thresholds:
        y_pred = np.where(y_scores >= threshold, 1, -1)
        current_f1 = f1_score(y_true, y_pred)
        current_acc = accuracy(y_true, y_pred)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold
            best_accuracy = current_acc

    return best_threshold, best_f1, best_accuracy

def optimize_thresholdlog(y_true, y_scores):
    """
    Find the optimal threshold maximizing F1-score for logistic regression outputs.

    Args:
        y_true (np.ndarray): True binary labels (0 or 1).
        y_scores (np.ndarray): Predicted probabilities from the logistic model.

    Returns:
        tuple: (best_threshold, best_f1, best_accuracy)
    """

    best_threshold = None
    best_f1 = 0
    best_accuracy = 0

    thresholds = np.linspace(np.min(y_scores), np.max(y_scores), 100)

    for threshold in thresholds:
        y_pred = np.where(y_scores >= threshold, 1, 0)
        current_f1 = f1_scorelog(y_true, y_pred)
        current_acc = accuracy(y_true, y_pred)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold
            best_accuracy = current_acc

    return best_threshold, best_f1, best_accuracy

def sigmoid(t):
    """
    Apply the sigmoid function on t.

    Parameters:
    t : numpy array
        Input data.

    Returns:
    s : numpy array
        The sigmoid of the input data.
    """
    t = np.where(t > 500, 500, t)
    t = np.where(t < -500, -500, t)
    return 1.0 / (1.0 + np.exp(-t))

def predict(y_scores, threshold):
    """
    Predict binary labels based on a decision threshold.

    Args:
        y_scores (np.ndarray): Continuous model scores or probabilities.
        threshold (float): Decision boundary for classification.

    Returns:
        np.ndarray: Predicted labels (1 or -1).
    """
    return np.where(y_scores >= threshold, 1, -1)


def build_k_indices(y, k_fold, seed):
    """
    Build indices for k-fold cross-validation.

    Args:
        y (np.ndarray): Labels array of shape (N,).
        k_fold (int): Number of folds (K) for cross-validation.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: 2D array of shape (k_fold, N/k_fold) containing the indices for each fold.
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def compute_mse(y, tx, w):
    """
    Compute the Mean Squared Error (MSE) loss.

    Args:
        y (np.ndarray): True target values of shape (N,).
        tx (np.ndarray): Input data matrix of shape (N, D).
        w (np.ndarray): Weight vector of shape (D,).

    Returns:
        float: MSE loss value.
    """
    e = y - tx.dot(w)
    return (e @ e) / (2 * len(y))

def cross_validation_gd(y, tx, k_fold, gammas, max_iters=1000, seed=1):
    """
    Perform k-fold cross-validation to select the best learning rate (gamma)
    for gradient descent based on F1 score.

    Args:
        y (np.ndarray): Target values of shape (N,).
        tx (np.ndarray): Input data matrix of shape (N, D).
        k_fold (int): Number of folds for cross-validation.
        gammas (list[float]): List of learning rates to evaluate.
        max_iters (int, optional): Maximum number of gradient descent iterations. Default is 1000.
        seed (int, optional): Random seed for reproducibility. Default is 1.

    Returns:
        float: The gamma value that yields the highest F1 score across folds.
    """
    k_indices = build_k_indices(y, k_fold, seed)
    avg_f1_per_gamma = []

    for gamma in gammas:
        f1_list = []
        for k in range(k_fold):
            te_idx = k_indices[k]
            tr_idx = np.hstack([k_indices[i] for i in range(k_fold) if i != k])
            y_tr, tx_tr = y[tr_idx], tx[tr_idx]
            y_te, tx_te = y[te_idx], tx[te_idx]

            initial_w = np.zeros(tx_tr.shape[1])
            w, _ = mean_squared_error_gd(y_tr, tx_tr, initial_w, max_iters, gamma)
            # Compute predicted scores (raw model outputs before applying threshold)
            y_scores = tx_te @ w

            # Find the best classification threshold based on F1 score
            _, best_f1_gd, = optimize_threshold(y_te, y_scores)
            
            f1_list.append(best_f1_gd)

        avg_f1 = np.mean(f1_list)
        avg_f1_per_gamma.append(avg_f1)
        print(f"Gamma = {gamma:.5f} | Avg F1 = {avg_f1:.5f}")

    best_gamma = gammas[np.argmax(avg_f1_per_gamma)]

    return  best_gamma

def cross_validation_sgd(y, tx, k_fold, gammas, max_iters=1000, seed=1, n_restarts=3):
    """
    Perform k-fold cross-validation to select the best learning rate (gamma)
    for stochastic gradient descent (SGD) based on F1-score

    Args:
        y (np.ndarray): Target values of shape (N,).
        tx (np.ndarray): Input data matrix of shape (N, D).
        k_fold (int): Number of folds for cross-validation.
        gammas (list[float]): List of learning rates to test.
        max_iters (int, optional): Maximum number of SGD iterations. Default is 1000.
        seed (int, optional): Random seed for reproducibility. Default is 1.
        n_restarts (int, optional): Number of restarts per fold. Default is 3.

    Returns:
        float: The gamma value that yields the highest F1-score across folds and restarts.
    """
    
    k_indices = build_k_indices(y, k_fold, seed)
    avg_f1_per_gamma = []

    for gamma in gammas:
        fold_f1 = []
        for k in range(k_fold):
            te_idx = k_indices[k]
            tr_idx = np.hstack([k_indices[i] for i in range(k_fold) if i != k])
            y_tr, tx_tr = y[tr_idx], tx[tr_idx]
            y_te, tx_te = y[te_idx], tx[te_idx]
            rest_f1 = []
            for r in range(n_restarts):
                np.random.seed(seed + 1000*k + r)  
                initial_w = np.zeros(tx_tr.shape[1])
                w, _ = mean_squared_error_sgd(y_tr, tx_tr, initial_w, max_iters, gamma)
                # Compute prediction scores on the test set
                y_scores = tx_te @ w

                # Find optimal classification threshold (maximizing F1 score)
                _, best_f1_sgd, _ = optimize_threshold(y_te, y_scores)
                rest_f1.append(best_f1_sgd)

            fold_f1.append(np.mean(rest_f1))

        avg_f1 = np.mean(fold_f1)
        avg_f1_per_gamma.append(avg_f1)
        print(f"[SGD] Gamma = {gamma:.5f} | Avg F1 = {avg_f1:.6f}")

    best_gamma = gammas[np.argmax(avg_f1_per_gamma)]
    print(f"=> Best gamma (SGD): {best_gamma} with Avg F1 = {max(avg_f1_per_gamma):.6f}")
    return best_gamma

def cross_validation_ridge(y, tx, k_fold, lambdas, seed=1):
    """
    Perform k-fold cross-validation to select the best regularization parameter (lambda)
    for ridge regression based on F1-score.

    Args:
        y (np.ndarray): Target values of shape (N,).
        tx (np.ndarray): Input data matrix of shape (N, D).
        k_fold (int): Number of folds for cross-validation.
        lambdas (list[float]): List of lambda values to test.
        seed (int, optional): Random seed for reproducibility. Default is 1.

    Returns:
        float: The lambda value that yields the highest average F1-score across folds.
    """
    k_indices = build_k_indices(y, k_fold, seed)
    avg_f1_per_lambda = []

    for lambda_ in lambdas:
        f1_list = []
        for k in range(k_fold):
            te_idx = k_indices[k]
            tr_idx = np.hstack([k_indices[i] for i in range(k_fold) if i != k])

            y_tr, tx_tr = y[tr_idx], tx[tr_idx]
            y_te, tx_te = y[te_idx], tx[te_idx]

            w, _ = ridge_regression(y_tr, tx_tr, lambda_)

            # mse_te = compute_mse(y_te, tx_te, w)
            # mse_list.append(mse_te)
            y_scores = tx_te @ w

            # Find optimal threshold (maximizing F1 score)
            _, best_f1_rr, _ = optimize_threshold(y_te, y_scores)
            f1_list.append(best_f1_rr)

        avg_f1 = np.mean(f1_list)
        avg_f1_per_lambda.append(avg_f1)
        print(f"[Ridge] lambda = {lambda_:.6f} | Avg f1 = {avg_f1:.6f}")

    best_lambda = lambdas[np.argmax(avg_f1_per_lambda)]
    print(f"=> Best lambda (Ridge): {best_lambda} with Avg F1 = {max(avg_f1_per_lambda):.6f}") 

    return best_lambda  



def cross_validation_logreg(y, tx, k_fold, gammas, max_iters=1000, seed=1,verbose=True):
    """
    Perform k-fold cross-validation to select the best learning rate (gamma)
    for logistic regression based on F1-score.

    Args:
        y (np.ndarray): True binary labels (0 or 1).
        tx (np.ndarray): Input data matrix of shape (N, D).
        k_fold (int): Number of folds for cross-validation.
        gammas (list[float]): List of learning rates to test.
        max_iters (int, optional): Maximum number of iterations. Default is 1000.
        seed (int, optional): Random seed for reproducibility. Default is 1.
        verbose (bool, optional): If True, prints results for each gamma. Default is True.

    Returns:
        float: The gamma value that yields the lowest average F1-score across folds.
    """
    k_indices = build_k_indices(y, k_fold, seed)
    avg_f1_per_gamma = []
    

    for gamma in gammas:
        fold_f1 = []
        
        for k in range(k_fold):
            te_idx = k_indices[k]
            tr_idx = np.hstack([k_indices[i] for i in range(k_fold) if i != k])

            y_tr, tx_tr = y[tr_idx], tx[tr_idx]
            y_te, tx_te = y[te_idx], tx[te_idx]

            w0 = np.zeros(tx_tr.shape[1], dtype=float)
            w, _ = logistic_regression(y_tr, tx_tr, w0, max_iters, gamma)

            # Compute predicted probabilities on the test set
            y_scores = sigmoid(tx_te @ w)

            # Find optimal threshold (maximizing F1 score)
            _, best_f1_lr, _ = optimize_thresholdlog(y_te, y_scores)
            

            fold_f1.append(best_f1_lr)
            

        avg_f1 = float(np.mean(fold_f1))

        avg_f1_per_gamma.append(avg_f1)
        
        print(f"gamma={gamma:.6g} | avg F1={avg_f1:.6f} ")

    best_idx = int(np.argmax(avg_f1_per_gamma))
    
    best_gamma = float(gammas[best_idx])

    print(f"=> Best gamma : {best_gamma} with Avg F1 = {max(avg_f1_per_gamma):.6f}")

    return best_gamma

def cross_validate_reg_logreg(
    y, tx, k_fold, lambdas, gammas,
    max_iters=1000, seed=1,):
    """
    Perform k-fold cross-validation to find the best combination of
    regularization strength (lambda) and learning rate (gamma)
    for regularized logistic regression, based on F1-score.

    Args:
        y (np.ndarray): True binary labels (0 or 1).
        tx (np.ndarray): Input data matrix of shape (N, D).
        k_fold (int): Number of folds for cross-validation.
        lambdas (list[float]): List of regularization strengths to test.
        gammas (list[float]): List of learning rates to test.
        max_iters (int, optional): Maximum number of iterations. Default is 1000.
        seed (int, optional): Random seed for reproducibility. Default is 1.

    Returns:
        tuple[float, float]: (best_lambda, best_gamma) — the parameters yielding the lowest average F1-score.
    """
    
    
    k_indices = build_k_indices(y, k_fold, seed)

    L = len(lambdas)
    G = len(gammas)
    avg_f1 = np.zeros((L, G), dtype=float)
    

    for i, lambda_ in enumerate(lambdas):
        for j, gamma in enumerate(gammas):
            fold_f1s = []
            

            for k in range(k_fold):
                te_idx = k_indices[k]
                tr_idx = np.concatenate([k_indices[t] for t in range(k_fold) if t != k])

                X_tr, y_tr = tx[tr_idx], y[tr_idx]
                X_te, y_te = tx[te_idx], y[te_idx]

                w0 = np.zeros(X_tr.shape[1], dtype=float)
                w, _ = reg_logistic_regression(y_tr, X_tr, lambda_, w0, max_iters, gamma)
                # Compute predicted probabilities on the test set
                y_scores = sigmoid(X_te @ w)

                # Find optimal threshold (maximizing F1 score)
                _, best_f1_reg, _= optimize_thresholdlog(y_te,y_scores)
                
                fold_f1s.append(best_f1_reg)

            avg_f1[i, j] = np.mean(fold_f1s)

            print(f"λ={lambda_:.3g}  γ={gamma:.3g}  | --- f1={avg_f1[i,j]:.6f}")

    best_flat = np.argmax(avg_f1)
    best_i, best_j = np.unravel_index(best_flat, avg_f1.shape)
    best_lambda = float(lambdas[best_i])
    best_gamma = float(gammas[best_j])

    
    print(f"=> Best: lambda={best_lambda}  gamma={best_gamma}  "
              f"with avg F1={avg_f1[best_i,best_j]:.6f} ")

   
    return best_lambda, best_gamma  
