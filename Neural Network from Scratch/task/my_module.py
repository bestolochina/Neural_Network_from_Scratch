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
        self.weights: np.ndarray = xavier(self.n_features, self.n_classes)

        # Xavier initialization for biases: shape (1, n_classes)
        # Note: This is task-specific; typical practice uses zeros.
        self.biases: np.ndarray = xavier(1, self.n_classes)

        # Will hold the output after forward pass
        self.output: np.ndarray | None = None

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
        self.output = a

        return self.output

