import numpy as np


def scale(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Scales the training and test datasets by normalizing features based on training data.

    Each feature in the training and test sets is divided by the corresponding feature's
    maximum value from the training set. To prevent division by zero, features with a
    maximum value of zero are replaced with 1 during scaling.

    Parameters:
        x_train (np.ndarray): Training dataset of shape (n_samples_train, n_features).
        x_test (np.ndarray): Test dataset of shape (n_samples_test, n_features).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the scaled training and test datasets.
    """
    max_values: np.ndarray = x_train.max(axis=0)
    max_values[max_values == 0] = 1  # to prevent division by 0 in case all values of the feature are 0

    x_train_rescaled = x_train / max_values
    x_test_rescaled = x_test / max_values
    return x_train_rescaled, x_test_rescaled


def xavier(n_in: int, n_out: int) -> np.ndarray:
    """
    Initializes a weight matrix using Xavier (Glorot) uniform initialization.

    The weights are drawn from a uniform distribution in the range
    [-sqrt(6 / (n_in + n_out)), sqrt(6 / (n_in + n_out))].

    Parameters:
        n_in (int): Number of input units.
        n_out (int): Number of output units.

    Returns:
        np.ndarray: Initialized weight matrix of shape (n_in, n_out).
    """
    limit = np.sqrt(6 / (n_in + n_out))
    weights = np.random.uniform(-limit, limit, (n_in, n_out))
    return weights


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """
    Applies the sigmoid activation function element-wise.

    The sigmoid function is defined as:
        sigmoid(x) = 1 / (1 + exp(-x))

    It maps input values to the range (0, 1), making it useful for binary classification
    and as an activation function in neural networks.

    Parameters:
        x (float or np.ndarray): Input value or array of values.

    Returns:
        float or np.ndarray: Sigmoid of the input, with the same shape as x.
    """
    return 1 / (1 + np.exp(-x))


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate Mean Squared Error (MSE) between predictions and true values.

    Args:
        y_pred (np.ndarray): Predicted values.
        y_true (np.ndarray): Ground truth values.

    Returns:
        float: The mean squared error.
    """
    return ((y_pred - y_true) ** 2).mean()


def mse_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Compute the gradient (derivative) of the Mean Squared Error with respect to predictions.

    Args:
        y_pred (np.ndarray): Predicted values.
        y_true (np.ndarray): Ground truth values.

    Returns:
        np.ndarray: Derivative of MSE with respect to y_pred.
    """
    return 2 * (y_pred - y_true)


def sigmoid_derivative(x: float | np.ndarray) -> float | np.ndarray:
    """
    Compute the derivative of the sigmoid function.

    Args:
        x (float | np.ndarray): Input value(s).

    Returns:
        float | np.ndarray: Derivative of the sigmoid at the given input(s).
    """
    sigma = sigmoid(x)
    return sigma * (1 - sigma)


class OneLayerNeural:
    def __init__(self, n_features: int, n_classes: int) -> None:
        """
        Initializes a one-layer neural network.

        Parameters:
            n_features (int): Number of input features (size of each input sample).
            n_classes (int): Number of output classes (number of neurons in output layer).

        Attributes:
            weights (np.ndarray): Weight matrix of shape (n_features, n_classes), initialized using Xavier.
            biases (np.ndarray): Bias vector of shape (1, n_classes), initialized using Xavier (non-standard).
            output (np.ndarray | None): Stores the result of the last forward pass.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # Xavier initialization for weights: shape (n_features, n_classes)
        # np.random.seed(2023)
        self.weights: np.ndarray = xavier(self.n_features, self.n_classes)

        # Xavier initialization for biases: shape (1, n_classes)
        # Note: This is task-specific; typical practice uses zeros.
        self.biases: np.ndarray = xavier(1, self.n_classes)

        # Will hold the output after forward pass
        self.output: np.ndarray | None = None
        self.Z: np.ndarray | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the neural network using sigmoid activation.

        Parameters:
            X (np.ndarray): Input data of shape (batch_size, n_features)

        Returns:
            np.ndarray: Output of the network after applying weights, biases, and activation.
                        Shape: (batch_size, n_classes)
        """
        # Linear transformation: Z = XW + b
        z = X @ self.weights + self.biases  # (batch_size, n_classes)

        # Non-linear activation using sigmoid
        a = sigmoid(z)

        # Store the result for potential use in backpropagation
        self.Z = X @ self.weights + self.biases  # Store Z
        self.output = sigmoid(self.Z)

        return self.output

    def backprop(self, X: np.ndarray, y_true: np.ndarray, alpha: float = 0.1):
        """
        Performs one step of backpropagation and updates the model's weights and biases.

        Parameters:
            X (np.ndarray): Input data of shape (batch_size, n_features).
            y_true (np.ndarray): True labels of shape (batch_size, n_classes).
            alpha (float): Learning rate for gradient descent. Default is 0.1.

        The method uses the mean squared error (MSE) loss and sigmoid activation.
        It assumes that self.output has already been computed via a forward pass.
        """
        y_pred = self.output
        error = mse_derivative(y_pred, y_true)
        sig_grad = sigmoid_derivative(self.Z)  # Use Z from the forward pass
        delta = error * sig_grad
        grad_w = X.T @ delta
        grad_b = np.sum(delta, axis=0, keepdims=True)

        self.weights -= alpha * grad_w
        self.biases -= alpha * grad_b


def epoch_training(estimator: OneLayerNeural, alpha: float,
                   X: np.ndarray, y: np.ndarray, batch_size = 100) -> float:
    indeces = np.random.permutation(len(X))
    X_shuffled = X[indeces]
    y_shuffled = y[indeces]

    n_bathes = len(X) // batch_size + (len(X) % batch_size != 0)
    for i in range(n_bathes):
        start = i * batch_size
        end = start + batch_size
        estimator.forward(X=X_shuffled[start: end])
        estimator.backprop(X=X_shuffled[start: end], y_true=y_shuffled[start: end], alpha=alpha)

    estimator.forward(X=X_shuffled[start: end])
    return mse(y_pred=estimator.output, y_true=y_shuffled[start: end])
