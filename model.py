from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.engine.saving import model_from_json
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import tensorflow as tf

def save_model(model, name):
    model_json = model.to_json()
    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name + ".h5")
    print("Saved model to disk")


def load_from_file(name):
    json_file = open(name + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name + ".h5")
    print("Loaded model from disk")

"""
Creates a model and then train it or load weights from filename.h5
"""
#TODO dense-dense-dense
def get_model(load_instead_fit, save, filename=None, epochs=None, x_train=None, y_train=None):

    input_shape = (28,28,1)

    model = Sequential()
    model.add(Flatten())
    model.add(Dense(28*28, activation=tf.nn.relu))
    model.add(Dropout(0.2))

    model.add(Dense(28*28, activation=tf.nn.relu))
    model.add(Dropout(0.2))

    model.add(Dense(28*28, activation=tf.nn.relu))
    model.add(Dropout(0.2))

    model.add(Dense(28*28, activation=tf.nn.relu))
    model.add(Dropout(0.2))


    model.add(Dense(100, activation=tf.nn.softmax))



    callbacks = [EarlyStopping(monitor='val_loss', patience=2)]

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    #model.summary()

    if load_instead_fit:
        model.load_weights(filename+".h5")
    else:
        model.fit(x_train[1000:], y_train[1000:],validation_data=(x_train[:1000], y_train[:1000]), epochs=epochs, callbacks=callbacks)
    if save: save_model(model, filename)
    return model

