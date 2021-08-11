import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from plotting import plot_learning_history


def deep_neural_network(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray = None, y_test: np.ndarray = None):
    """
    Deep Neural Network created for classification input data.

    References
    ----------
    [1] https://keras.io/

    Parameters
    ----------
    X_train : ndarray
        Array of input data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ].

    y_train : ndarray
        Array of labels belongs to input X_train data.

    X_test : ndarray {default: None}
        Array of data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ]. Could be None.

    y_test : ndarray {default: None}
        Array of labels belongs to X_test data. Could be None.

    Returns
    -------
    dnn
        Trained classifier ready to predict.
    """
    # _check_deep_network_params(X_train, y_train)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2,
                                                      random_state=1, stratify=y_train)

    tf.random.set_seed(1)
    num_epochs = 95
    batch_size = 100
    steps_per_epoch = int(np.ceil(len(y_train) / batch_size))

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=32,
                                    activation='tanh'))
    model.add(tf.keras.layers.Dense(units=256,
                                    activation='softsign'))
    model.add(tf.keras.layers.Dense(units=512,
                                    activation='tanh'))
    model.add(tf.keras.layers.Dense(units=256,
                                    activation='softsign'))
    model.add(tf.keras.layers.Dense(units=64,
                                    activation='softplus'))
    model.add(tf.keras.layers.Dense(units=1,
                                    activation='sigmoid'))
    model.build(input_shape=(None, len(X_train[0])))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['accuracy'])

    hist = model.fit(X_train, y_train,
                     batch_size=batch_size,
                     epochs=num_epochs,
                     steps_per_epoch=steps_per_epoch,
                     validation_data=(X_val, y_val))
    plot_learning_history(hist.history)

    if X_test is not None and y_test is not None and len(X_test) == len(y_test):
        y_pred = model.predict(X_test)
        y_pred = list(map(lambda item: 0 if item[0] <= 0.5 else 1, y_pred))
        print(sum(y_pred), len(y_pred))
        print(f"Deep Neural Network test accuracy: {accuracy_score(y_test, y_pred)}")

    return model


# TODO
def _check_deep_network_params(X: np.ndarray, y: np.ndarray):
    """
    Check the number of layers, types of layers, activation functions etc.
    needed in a deep neural network. Show results on plots.

    Parameters
    ----------
    X : ndarray
        Array of input data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ].

    y : ndarray
        Array of labels belongs to input X data.
    """
    pass


def train_input_fn(X_train, y_train, batch_size=10):
    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(X_train), y_train))
    return dataset.shuffle(10000).repeat().batch(batch_size)
