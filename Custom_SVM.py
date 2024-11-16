import numpy as np


class Custom_SVM(object):
    def __init__(self, C=1.0, eta=0.01, num_iterations=1000):
        """
        Initializes the SVM classifier with the given parameters.

        Parameters:
        - C: Regularization parameter (default is 1.0).
        - eta: Learning rate (default is 0.01).
        - num_iterations: Number of gradient descent iterations (default is 1000).
        """
        self.C = C
        self.eta = eta
        self.num_iterations = num_iterations
        self.W = None  # Weights vector (W, b) will be stored here
        self.test_accuracy = list()
        self.test_loss = list()
        self.train_accuracy = list()
        self.train_loss = list()

    def fit(self, X, y, X_test = None, y_test = None, verbose = False):
        """
        Trains the SVM classifier using gradient descent.

        Parameters:
        - X: numpy array of shape (m, n), where m is the number of training samples and n is the number of features.
        - y: numpy array of shape (m,), where each element is the label (+1 or -1).
        """

        m, n = X.shape

        # Augment the input matrix X to include a column of ones for the bias term
        X_augmented = np.hstack([X, np.ones((m, 1))])  # Shape (m, n+1)

        # Initialize the combined weight vector (W, b) with small random values
        self.W_combined = np.random.randn(n + 1) * 0.01  # Shape (n+1,)

        # Gradient descent loop
        for iteration in range(self.num_iterations):
            # Compute the decision function for all training samples
            decision_values = np.dot(X_augmented, self.W_combined)  # Shape (m,)

            # Compute the hinge loss gradients
            hinge_loss_gradient = np.zeros_like(self.W_combined)

            for i in range(m):
                if y[i] * decision_values[i] < 1:
                    hinge_loss_gradient -= self.C * y[i] * X_augmented[i]

            # The regularization gradient is simply the weight vector (including bias term)
            regularization_gradient = self.W_combined

            # Total gradient for the current iteration
            gradient = regularization_gradient + hinge_loss_gradient

            # Update the combined weight vector (W, b) using gradient descent
            self.W_combined -= self.eta * gradient

            # Compute the loss
            loss = self.compute_loss(X, y)

            # Compute accuracy
            accuracy = self.compute_accuracy(X, y)

            self.train_accuracy.append(accuracy)
            self.train_loss.append(loss)

            # Optionally: Print the progress every 100 iterations
            if X_test is not None and y_test is not None:
                test_loss = self.compute_loss(X_test, y_test)
                test_acc = self.compute_accuracy(X_test, y_test)
                self.test_loss.append(test_loss)
                self.test_accuracy.append(test_acc)
                # print(f"Iteration {iteration}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

        # After training, separate W and b from the combined weight vector
        self.W = self.W_combined[:-1]  # Weight vector W
        self.b = self.W_combined[-1]  # Bias term b

    def compute_loss(self, X, y):
        """
        Computes the total loss (hinge loss + regularization) for the current model.

        Parameters:
        - X: Input data of shape (m, n)
        - y: True labels of shape (m,)

        Returns:
        - Total loss (scalar)
        """
        m = X.shape[0]
        X_augmented = np.hstack([X, np.ones((m, 1))])  # Augment X to include the bias term

        # Compute the decision function (W * X + b)
        decision_values = np.dot(X_augmented, self.W_combined)  # Shape (m,)

        # Compute hinge loss
        hinge_loss = np.maximum(0, 1 - y * decision_values)

        # Regularization term (1/2) * ||W||^2
        regularization_loss = 0.5 * np.sum(self.W_combined[:-1] ** 2)  # Exclude the bias term from the regularization

        # Total loss: hinge loss + regularization
        total_loss = regularization_loss + self.C * np.sum(hinge_loss)

        return total_loss

    def compute_accuracy(self, X, y):
        """
        Computes the accuracy of the model on the training set.

        Parameters:
        - X: Input data of shape (m, n)
        - y: True labels of shape (m,)

        Returns:
        - Accuracy as a float
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

    def predict(self, X):
        """
        Predicts the class labels for the given input samples.

        Parameters:
        - X: numpy array of shape (m, n), where m is the number of samples to classify and n is the number of features.

        Returns:
        - predictions: numpy array of shape (m,), where each element is the predicted label (+1 or -1).
        """
        m = X.shape[0]

        # Augment the input matrix X to include a column of ones for the bias term
        X_augmented = np.hstack([X, np.ones((m, 1))])  # Shape (m, n+1)

        # Compute the decision function for each sample
        decision_values = np.dot(X_augmented, self.W_combined)  # Shape (m,)

        # Predict the labels (+1 or -1) based on the decision function
        predictions = np.sign(decision_values)  # Classify based on the sign of decision value

        return predictions

    def predict_proba(self, x):
        """
        Predict class probabilities using the trained SVM model.

        Parameters:
        - X: Feature matrix (N x d)

        Returns:
        - probabilities: Probabilities of each class (-1 or +1) for each sample
        """
        # Add a column of ones to X to account for the bias term (augmented X)
        X_augmented = np.hstack([X, np.ones((m, 1))])

        # Compute the decision function
        decision_values = np.dot(X_augmented, self.W_combined)

        # Apply the sigmoid function to get probabilities in range [0, 1]
        prob = 1 / (1 + np.exp(-decision_values))  # Probability for class +1

        return np.column_stack([1 - prob, prob])

    def get_params(self):
        """
        Get the learned parameters of the model (bias and weights).

        Returns:
        - w_star: Augmented weight vector [b, w]
        """
        return self.W, self.b
    def get_history(self):

        return self.train_accuracy, self.test_accuracy, self.train_loss, self.test_loss