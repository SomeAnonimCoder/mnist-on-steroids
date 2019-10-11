import numpy as np
import tensorflow as tf

def get_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    print(x_train.shape)
    print(y_train.shape)
    input_shape = (28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    return x_train,y_train, x_test, y_test

def mixed_pairs(size, x, y):
    indexes = np.random.randint(0, y.shape[0]- 1, (size, 2))

    x_ = []
    y_ = []

    for i in indexes:
        a = i[0]
        b = i[1]
        x_.append(np.clip(np.add(x[a], x[b]), 255), 0, 255)
        y_.append(min(y[a], y[b]) + max(y[b], y[a]) * 10)

    x = np.asarray(x_) / 255 / 2
    y= np.asarray(y_)
    return x,y
