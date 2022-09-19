# Machine-Learning

Implementing modules to apply linear and non-linear prediction models on practical and random datasets.

Modules:

1) my_knn.py

Implements a simple function that performs KNN classification and regression.

2) linear_regression.py

Implements different linear regression model families and subset selections including OLS, Lasso, Ridge, Adaptive Lasso, Elastic Net, Best Subsets, and Forward/Backward selection models and makes a comparisen between their performance.

3) linear_regression_cv.py

Implemetns all the models in the linear_regression.py module by using cross validation. It splits the dataset into 60-20-20% as training, validation, and test sets, then trains the model using training set, tests all the model family (sweeping hyperparameters) on the validation set to find the best values for hyperparameters that minimize the test MSE, and eventually tests the model on test set. It repeats this process 10 times and measures the average test MSE.

4) lm_empirical_demonstrations.py

Implements a module that empirically demonstrates: centring dataset is equivalent with adding a column of 1's to it for removing the intercept term, OLS has zero training error when p>n, and MSE existence theorem. 
