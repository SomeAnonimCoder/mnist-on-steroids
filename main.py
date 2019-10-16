import matplotlib.pyplot as plt
from dataset import get_data, mixed_pairs
from model import get_model
from show_results import test_model


def __main__():
    x_train, y_train, x_test, y_test = get_data()
    SIZE = 100000
    # CREATE MIXED TRAIN

    x_train, y_train = mixed_pairs(SIZE, x_train, y_train)

    # CREATE MIXED TEST
    SIZE = 100000
    x_test, y_test = mixed_pairs(SIZE, x_test, y_test)


    model = get_model(
        filename="1",
        load_instead_fit=False,
        save=True,
        epochs=5,
        x_train=x_train,
        y_train=y_train,
    )

    model.summary()

    print(test_model(model, x_test, y_test))

if __name__=="__main__":
    __main__()
