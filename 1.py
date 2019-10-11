import random

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.engine.saving import model_from_json


def show(img):
    plt.imshow(img.reshape(28, 28))
    plt.show()

def save(model, name):
    model_json = model.to_json()
    with open(name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name+".h5")
    print("Saved model to disk")

def load(name):
    json_file = open(name+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name+".h5")
    print("Loaded model from disk")


SIZE = 100000
#CREATE MIXED TRAIN



#CREATE MIXED TEST
SIZE = 1000
indexes = np.random.randint(0, 10000-1, (SIZE,2))

x_test_ = []
y_test_ = []

for i in indexes:
    a = i[0]
    b = i[1]
    x_test_.append(np.add(x_test[a], x_test[b]))
    y_test_.append(min(y_test[a], y_test[b])+max(y_test[b], y_test[a])*10)

x_test = np.asarray(x_test_)/255/2
y_test = np.asarray(y_test_)


model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(100,activation=tf.nn.softmax))


callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.load_weights("1.h5")

#model.fit(x_train,y_train, epochs=5, callbacks=callbacks)
#save(model, "1")

print(model.evaluate(x_test, y_test))
print(model.metrics_names)

# for i in [53,128,569,463]:
#     image = x_test[i].reshape(28,28)
#     answer = (y_test[i])
#     print("ans: {}".format(answer))
#     plt.imshow(image, cmap="gray")
#     plt.show()
#     pred = model.predict(image.reshape(1, 28, 28, 1))
#     print("cnn says: {}".format(pred.argmax()))



for i in range(0,10000-1):
    res = model.predict(x_test[i].reshape(1,28,28,1))
    if(res.argmax() != y_test[i]):
        plt.imsave("said{}real{}.png".format(res.argmax(), y_test[i]), x_test[i].reshape(28, 28))

        print()

