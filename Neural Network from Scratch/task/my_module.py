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


class OneLayerNeural:
    def __init__(self, n_features: int, n_classes: int) -> None:
        # Initiate weights and biases using Xavier
        self.n_features = n_features
        self.n_classes = n_classes
        self.weights: np.ndarray = xavier(self.n_features, self.n_classes)
        self.biases: np.ndarray = xavier(1, self.n_classes)  # np.zeros((n_classes,))
        self.output: np.ndarray | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.weights + self.biases
        a = sigmoid(z)
        self.output = a
        return self.output
