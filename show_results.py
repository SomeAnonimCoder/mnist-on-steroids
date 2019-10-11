import matplotlib.pyplot as plt

def show_pic(img):
    plt.imshow(img.reshape(28, 28))
    plt.show()


def test_model(model, x_test, y_test):
    log = ""
    j=0
    for i in range(0, y_test.shape[0] - 1):
        res = model.predict(x_test[i].reshape(1, 28, 28, 1))
        if (res.argmax() != y_test[i]):
            plt.imsave("output/said{}real{}.png".format(res.argmax(), y_test[i]), x_test[i].reshape(28, 28), cmap="gray")
            j+=1
            log+="""number: {} predicted: {} real {}\n""".format(j,res.argmax(),y_test[i])
    file = open("output/log.txt", "w")
    file.write(log)
    file.close()
    return model.evaluate(x_test, y_test)