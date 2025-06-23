import numpy as np
import pandas as pd
import os
import requests
from matplotlib import pyplot as plt
import my_module
from tqdm import tqdm
from utils.utils import custom_uniform

np.random.uniform = custom_uniform


# scroll to the bottom to start coding your solution


def one_hot(data: np.ndarray) -> np.ndarray:
    y_train = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y_train[rows, data] = 1
    return y_train


def plot(loss_history: list, accuracy_history: list, filename='plot'):

    # function to visualize learning process at stage 4

    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')


if __name__ == '__main__':

    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if ('fashion-mnist_train.csv' not in os.listdir('../Data') and
            'fashion-mnist_test.csv' not in os.listdir('../Data')):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_train.csv', 'wb').write(r.content)
        print('Loaded.')

        print('Test dataset loading.')
        url = "https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_test.csv', 'wb').write(r.content)
        print('Loaded.')

    # Read train, test data.
    raw_train = pd.read_csv('../Data/fashion-mnist_train.csv')
    raw_test = pd.read_csv('../Data/fashion-mnist_test.csv')

    X_train = raw_train[raw_train.columns[1:]].values
    X_test = raw_test[raw_test.columns[1:]].values

    y_train = one_hot(raw_train['label'].values)
    y_test = one_hot(raw_test['label'].values)

    # write your code here

    # Use scale to rescale X_train and X_test;
    X_train_rescaled, X_test_rescaled = my_module.scale(X_train, X_test)
    n_samples, n_features = X_train_rescaled.shape
    n_classes = y_train.shape[1]

    model = my_module.OneLayerNeural(n_features=n_features, n_classes=n_classes)

    initial_accuracy = [my_module.accuracy(model, X=X_test_rescaled, y_true=y_test)]

    loss_logging = []
    accuracy_logging = []
    for epoch in range(20):
        loss = my_module.epoch_training(estimator=model, alpha=0.5,
                                        X=X_train_rescaled, y=y_train, batch_size=100)
        accuracy = my_module.accuracy(model, X=X_test_rescaled, y_true=y_test)

        loss_logging.append(loss)
        accuracy_logging.append(accuracy)

    print(initial_accuracy, accuracy_logging)

    plot(loss_history=loss_logging, accuracy_history=accuracy_logging)

