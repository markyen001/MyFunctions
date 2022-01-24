# Create a class that will hold all the functions.
# The functions include TrainTest, PolynomialRegression, and ExponentialRegression.
class RegressionModel:
    
    # Define the initialize method.
    # For a 2D dataset, the inputs will simply be the x-axis and the y-axis
    def __init__(self, x_axis, y_axis):
        # Set the x-axis
        self.x_axis = x_axis
        # Set the y-axis
        self.y_axis = y_axis
    
    # Define a function within this class to separate the data into training and testing data.
    # The additional input will be the desired training size expressed as a fraction.
    # This function does NOT randomize the data.
    def TrainTest(self,train_size_fraction):
        # Measure the length of the x-axis.
        x_length = self.x_axis.shape[0]
        
        # Get the x training data and the x testing data.
        # After multiplying the fraction with the length of the x-axis, round the value up.
        # Then convert the value from a float to an integer.
        self.x_train = self.x_axis[0:np.int_(np.ceil(x_length*train_size_fraction))]
        # The x testing data will take the remaining values.
        self.x_test = self.x_axis[np.int_(np.ceil(x_length*train_size_fraction)):]
        
        # Get the y training data and the y testing data.
        self.y_train = self.y_axis[0:np.int_(np.ceil(x_length*train_size_fraction))]
        self.y_test = self.y_axis[np.int_(np.ceil(x_length*train_size_fraction)):]
        return self.x_train, self.x_test, self.y_train, self.y_test
    
    # Define a function within this class to calculate the polynomial regression with
    # M-th order polynomial.
    # Polynomial regression has the form y = w0 + w1*x + w2*x**2 + ... + wM*x**M
    def PolynomialRegression(self, M):
        # Create the feature matrix, big X, for the training data.
        # For each x training data point, bring it to the m-th power from 0 to M.
        # Transpose it so that rows represent samples and columns represent features.
        X_train = np.array([self.x_train**m for m in range(M+1)]).T
        
        # Compute the weights by using the Moore–Penrose pseudoinverse.
        # Only works for small datasets.
        # Use the @ symbol to multiply matrices. Using * symbol gives different results.
        self.w_poly = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ self.y_train
        
        # Calculate the predicted y training values.
        self.y_train_predicted = X_train @ self.w_poly
        
        # Calculate the error for the y training values.
        self.error_train = self.y_train - self.y_train_predicted
        
        # Calculate the predicted y test values.
        # Need to first create the feature matrix, big X, for the test data.
        X_test = np.array([self.x_test**m for m in range(M+1)]).T
        # Then use the new feature matrix to calculate the predictions for the y test values.
        self.y_test_predicted = X_test @ self.w_poly
        
        # Calculate the error for the y test values.
        self.error_test = self.y_test - self.y_test_predicted
        
        # Output the values. If you don't want to save every variable, then use underscore _. Similar to
        # MATLAB's ~.
        return self.w_poly, self.y_train_predicted, self.error_train, self.y_test_predicted, self.error_test
    
    # Define a function within this class to use the Polynomial Regression function to extrapolate to
    # "future" data.
    def PolyRegExtrapolate(self, M, x_future):
        # The input x_future should be a numpy array.
        # Create the feature matrix, big X.
        X_future = np.array([x_future**m for m in range(M+1)]).T
        # Then use the feature matrix to calculate the extrapolated y value.
        self.y_future = X_future @ self.w_poly
        
        return self.y_future
    
    # Define a function within this class to calculate the exponential regression with
    # M-th order polynomial as the power of the exponential.
    # Exponential regression has the form y = exp(w0 + w1*x + w2*x**2 + ... + wM*x**M)
    def ExponentialRegression(self, M):
        # Create the feature matrix, big X, for the training data.
        # For each x training data point, bring it to the m-th power from 0 to M.
        # Transpose it so that rows represent samples and columns represent features.
        X_train = np.array([self.x_train**m for m in range(M+1)]).T
        
        # Compute the weights by using the Moore–Penrose pseudoinverse.
        # Only works for small datasets.
        # Use the @ symbol to multiply matrices. Using * symbol gives different results.
        # For the exponential regression, take the natural log of the y training values.
        self.w_exp = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ np.log(self.y_train)
        
        # Calculate the predicted y training values.
        self.y_train_predicted = np.exp(X_train @ self.w_exp)
        
        # Calculate the error for the y training values.
        self.error_train = self.y_train - self.y_train_predicted
        
        # Calculate the predicted y test values.
        # Need to first create the feature matrix, big X, for the test data.
        X_test = np.array([self.x_test**m for m in range(M+1)]).T
        # Then use the new feature matrix to calculate the predictions for the y test values.
        self.y_test_predicted = np.exp(X_test @ self.w_exp)
        
        # Calculate the error for the y test values.
        self.error_test = self.y_test - self.y_test_predicted
        
        # Output the values. If you don't want to save every variable, then use underscore _. Similar to
        # MATLAB's ~.
        return self.w_exp, self.y_train_predicted, self.error_train, self.y_test_predicted, self.error_test
    
    # Define a function within this class to use the Exponential Regression function to extrapolate to
    # "future" data.
    def ExpRegExtrapolate(self, M, x_future):
        # The input x_future should be a numpy array.
        # Create the feature matrix, big X.
        X_future = np.array([x_future**m for m in range(M+1)]).T
        # Then use the feature matrix to calculate the extrapolated y value.
        self.y_future = np.exp(X_future @ self.w_exp)
        
        return self.y_future