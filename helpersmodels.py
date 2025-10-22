from implementations import * 

def split_train(X, y, test_size=0.2, random_state=None):
    """Split the data into train and test sets."""
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
    Args:
        y_true (np.ndarray): True labels (0/1 or -1/1).
        y_pred (np.ndarray): Predicted labels (same shape).
    Returns:
        float: Accuracy (between 0 and 1).
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
    # True Positives (TP): correctly predicted positives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    # False Positives (FP): predicted positive but actually negative
    fp = np.sum((y_true == -1) & (y_pred == 1))
    # False Negatives (FN): predicted negative but actually positive
    fn = np.sum((y_true == 1) & (y_pred == -1))

    # Avoid division by zero
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    if (precision + recall) == 0:
        return 0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
def optimize_threshold(y_true, y_scores):
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



def sigmoid(z):
    """Compute the sigmoid function in a numerically stable way."""

    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z)) 

def predict(y_scores, threshold):
    """
    Predict binary labels using logistic regression parameters.
    Args:
        tx (np.ndarray): Data matrix of shape (N, D)
        w (np.ndarray): Weight vector of shape (D,)
        threshold (float): Decision threshold (default 0.5)
    Returns:
        np.ndarray: Predicted labels (0 or 1) of shape (N,)
    """
    return np.where(y_scores >= threshold, 1, -1)


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed
    
    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def compute_mse(y, tx, w):
    e = y - tx.dot(w)
    return (e @ e) / (2 * len(y))

def cross_validation_gd(y, tx, k_fold, gammas, max_iters=1000, seed=1):
    k_indices = build_k_indices(y, k_fold, seed)
    avg_mse_per_gamma = []

    for gamma in gammas:
        mse_list = []
        for k in range(k_fold):
            te_idx = k_indices[k]
            tr_idx = np.hstack([k_indices[i] for i in range(k_fold) if i != k])
            y_tr, tx_tr = y[tr_idx], tx[tr_idx]
            y_te, tx_te = y[te_idx], tx[te_idx]

            initial_w = np.zeros(tx.shape[1])
            w, _ = mean_squared_error_gd(y_tr, tx_tr, initial_w, max_iters, gamma)
            mse_te = compute_mse(y_te, tx_te, w)
            mse_list.append(mse_te)

        avg_mse = np.mean(mse_list)
        avg_mse_per_gamma.append(avg_mse)
        print(f"Gamma = {gamma:.5f} | Avg MSE = {avg_mse:.5f}")

    best_gamma = gammas[np.argmin(avg_mse_per_gamma)]

    return  best_gamma

def cross_validation_sgd(y, tx, k_fold, gammas, max_iters=1000, seed=1, n_restarts=3):
    """
    K-fold CV to pick the best gamma for SGD-based linear regression.

    Args:
        y, tx: dataset
        k_fold (int): number of folds
        gammas (iterable): learning rates to try
        max_iters (int): SGD updates per training run
        seed (int): seed for fold construction and restarts
        n_restarts (int): repeat SGD per fold to reduce randomness

    Returns:
        best_gamma (float)
    """
    k_indices = build_k_indices(y, k_fold, seed)
    avg_mse_per_gamma = []

    for gamma in gammas:
        fold_mses = []
        for k in range(k_fold):
            te_idx = k_indices[k]
            tr_idx = np.hstack([k_indices[i] for i in range(k_fold) if i != k])
            y_tr, tx_tr = y[tr_idx], tx[tr_idx]
            y_te, tx_te = y[te_idx], tx[te_idx]

            # Run SGD multiple times to reduce variance, average the fold MSE
            rest_mses = []
            for r in range(n_restarts):
                np.random.seed(seed + 1000*k + r)  # keep runs reproducible but different
                initial_w = np.zeros(tx.shape[1])
                w, _ = mean_squared_error_sgd(y_tr, tx_tr, initial_w, max_iters, gamma)
                mse_te = compute_mse(y_te, tx_te, w)
                rest_mses.append(mse_te)

            fold_mses.append(np.mean(rest_mses))

        avg_mse = np.mean(fold_mses)
        avg_mse_per_gamma.append(avg_mse)
        print(f"[SGD] Gamma = {gamma:.5f} | Avg MSE = {avg_mse:.6f}")

    best_gamma = gammas[np.argmin(avg_mse_per_gamma)]
    print(f"=> Best gamma (SGD): {best_gamma} with Avg MSE = {min(avg_mse_per_gamma):.6f}")
    return best_gamma

def cross_validation_ridge(y, tx, k_fold, lambdas, seed=1):
    """
    K-fold cross-validation to select the best lambda for ridge_regression.
    Prints Avg MSE per lambda and returns the best lambda.
    """
    k_indices = build_k_indices(y, k_fold, seed)
    avg_mse_per_lambda = []

    for lambda_ in lambdas:
        mse_list = []
        for k in range(k_fold):
            te_idx = k_indices[k]
            tr_idx = np.hstack([k_indices[i] for i in range(k_fold) if i != k])

            y_tr, tx_tr = y[tr_idx], tx[tr_idx]
            y_te, tx_te = y[te_idx], tx[te_idx]

            w, _ = ridge_regression(y_tr, tx_tr, lambda_)

            mse_te = compute_mse(y_te, tx_te, w)
            mse_list.append(mse_te)

        avg_mse = np.mean(mse_list)
        avg_mse_per_lambda.append(avg_mse)
        print(f"[Ridge] lambda = {lambda_:.6f} | Avg MSE = {avg_mse:.6f}")

    best_lambda = lambdas[np.argmin(avg_mse_per_lambda)]
    print(f"=> Best lambda (Ridge): {best_lambda} with Avg MSE = {min(avg_mse_per_lambda):.6f}")
    return best_lambda


