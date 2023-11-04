import numpy as np

def load_dataset():
    with np.load("mnist.npz") as f:
        x_train = f['x_train'].astype("float32") / 255
        x_train = np.reshape(x_train, (x_train.shape[0], -1))

        y_train = f['y_train']
        y_train = np.eye(10)[y_train]

    return x_train, y_train
