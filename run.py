import numpy as np
from helpers import *
from implementations import *
from helpersmodels import *

# Set the best hyperparameters 
best_lambda = 0.001
best_gamma = 0.3
best_threshold = 0.26567917946228264


# Get the current directory (the project root)
PROJECT_ROOT = os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# Load data
x_train = np.genfromtxt(
        os.path.join(DATA_DIR, "x_train_cleaned.csv"), delimiter=",", skip_header=1
    )
y_train = np.genfromtxt(
        os.path.join(DATA_DIR, "y_train_up.csv"), delimiter=",", skip_header=1
    )
# Split
X_train, X_test, y_train, y_test=split_train(x_train,y_train) 

#load test data
x_test_real= np.genfromtxt(
        os.path.join(DATA_DIR, "x_test_cleaned.csv"), delimiter=",", skip_header=1
    )
x_test_ids= np.genfromtxt(
        os.path.join(DATA_DIR, "x_test.csv"), delimiter=",",skip_header=1
    )
x_test_ids=x_test_ids[:, 0]

# Convert labels from {-1, 1} to {0, 1}
y_test = (y_test + 1) // 2
y_train = (y_train + 1) // 2

best_w, _ = reg_logistic_regression(y_train, X_train, 0.0001,  np.zeros(X_train.shape[1]), max_iters=1000, gamma=0.3)
y_scores = sigmoid(X_test @ best_w)
best_threshold, best_f1, best_accuracy=optimize_thresholdlog(y_test,y_scores)
y_predicted=predict(y_scores,best_threshold) 

# Compute predicted probabilities on the real test set
y_test_scores = sigmoid(x_test_real @ best_w)

# Predict labels using the optimal threshold
y_predicted = predict(y_test_scores, best_threshold)

# Generate submission file for the regularized logistic regression model
create_csv_submission(x_test_ids, y_predicted, "predRegLogisticReg")
