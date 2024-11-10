import numpy as np

def add_bias_feature(a):
    a_extended = np.zeros((a.shape[0],a.shape[1]+1))
    a_extended[:,:-1] = a
    a_extended[:,-1] = int(1)
    return a_extended

class CustomSVM(object):

    __class__ = "CustomSVM"
    __doc__ = """
    This is an implementation of the SVM classification algorithm
    Note that it works only for binary classification

    etha: float(default - 0.01)
        Learning rate, gradient step

    alpha: float, (default - 0.1)
        Regularization parameter in 0.5*alpha*||w||^2

    epochs: int, (default - 200)
        Number of epochs of training

    """

    def __init__(self, etha=0.01, alpha=0.1, epochs=200):
        self._epochs = epochs
        self._etha = etha
        self._alpha = alpha
        self._w = None
        self.history_w = list()
        self.train_errors = None
        self.val_errors = None
        self.train_loss = None
        self.val_loss = None

    def fit(self, X, y): #arrays: X; Y =-1,1
        """
        Train the SVM model using gradient descent.

        Parameters:
        - X: Feature matrix (N x d)
        - y: Labels vector (N,)

        Returns:
        - self: Trained SVM model
        """
        if len(set(y)) != 2:
            raise ValueError("Number of classes in Y is not equal 2!")

        X = add_bias_feature(X)
        self._w = np.random.normal(loc=0, scale=0.05, size=X.shape[1])
        self.history_w.append(self._w)
        train_errors = []
        train_loss_epoch = []

        for epoch in range(self._epochs):
            tr_err = 0
            tr_loss = 0
            for i,x in enumerate(X):
                margin = y[i]*np.dot(self._w,X[i])
                if margin >= 1:
                    self._w = self._w - self._etha*self._alpha*self._w
                    tr_loss += self.soft_margin_loss(X[i],y[i])
                else:
                    self._w = self._w +\
                    self._etha*(y[i]*X[i] - self._alpha*self._w)
                    tr_err += 1
                    tr_loss += self.soft_margin_loss(X[i],y[i])
                self.history_w.append(self._w)
            train_errors.append(tr_err)
            train_loss_epoch.append(tr_loss)
        self.history_w = np.array(self.history_w)
        self.train_errors = np.array(train_errors)
        self.train_loss = np.array(train_loss_epoch)

    def predict(self, X:np.array) -> np.array:
        y_pred = []
        X_extended = add_bias_feature(X)
        for i in range(len(X_extended)):
            y_pred.append(np.sign(np.dot(self._w,X_extended[i])))
        return np.array(y_pred)

    def hinge_loss(self, x, y):
        return max(0,1 - y*np.dot(x, self._w))

    def soft_margin_loss(self, x, y):
        return self.hinge_loss(x,y)+self._alpha*np.dot(self._w, self._w)

    def predict_proba(self, x):
        """
        Predict class probabilities using the trained SVM model.

        Parameters:
        - X: Feature matrix (N x d)

        Returns:
        - probabilities: Probabilities of each class (-1 or +1) for each sample
        """
        # Add a column of ones to X to account for the bias term (augmented X)
        X_extended = add_bias_feature(x)

        # Compute the decision function
        decision = np.dot(X_extended, self._w)  # (N,)

        # Apply the sigmoid function to get probabilities in range [0, 1]
        prob = 1 / (1 + np.exp(-decision))  # Probability for class +1

        return np.column_stack([1 - prob, prob])

    def get_params(self):
        """
        Get the learned parameters of the model (bias and weights).

        Returns:
        - w_star: Augmented weight vector [b, w]
        """
        return self._w[0], self._w[1:]