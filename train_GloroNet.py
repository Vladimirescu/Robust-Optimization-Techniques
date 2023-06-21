import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
import numpy as np
import os
from gloro import GloroNet
from gloro.training.losses import Crossentropy
from gloro.training.callbacks import UpdatePowerIterates
from models import model_AT_mnist, model_AT_cifar, res_net9, celeba_model
from utils import load_dataset


class GloroSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, path=""):
        self.path = path
        self.prev_best_loss = np.inf

    def on_epoch_end(self, epoch, logs):
        extracted_model = tf.keras.models.Model(inputs=self.model.inputs[0], outputs=self.model.layers[-2].output)
        if logs["val_loss"] < self.prev_best_loss:
            extracted_model.save_weights(self.path)
            current_loss = logs["val_loss"]
            print(f"Model improved from {self.prev_best_loss} to {current_loss} val_loss.")
            self.prev_best_loss = current_loss


datasets = {
    1: "MNIST",
    2: "FMNIST",
    3: "CIFAR-10",
    4: "eurosat",
    5: "CelebA"
}


if __name__ == "__main__":

    ###
    batch_size = 256
    dataset_type = 5
    ###

    dset_name = datasets[dataset_type]

    x_train, y_train, x_test, y_test, input_shape, n_classes = load_dataset(dataset_type)

    if dataset_type in [4, 5]:
        epochs = 200
        batch_size = 128

    for epsilon in [0.1, 0.2, 0.3]:
        model_save_path = f"models_GloroNet/{datasets[dataset_type]}/epsilon={epsilon}/"
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if dataset_type in [1, 2]:
            epochs = 30
            model = model_AT_mnist(input_shape=input_shape, n_classes=n_classes, batch_norm=False)
            model = GloroNet(model=model, epsilon=epsilon)
        elif dataset_type == 3:
            epochs = 100
            model = model_AT_cifar(input_shape=input_shape, n_classes=n_classes, batch_norm=False)
            model = GloroNet(model=model, epsilon=epsilon)
        elif dataset_type == 4:
            epochs = 100
            model = res_net9(input_shape=input_shape, n_classes=n_classes, deel=True)
            model.load_weights("models_GloroNet/eurosat/epsilon=500.0/model.hdf5")
            model = GloroNet(model=model, epsilon=epsilon)
        elif dataset_type == 5:
            epochs = 100
            model = celeba_model(input_shape, n_classes)
            model = GloroNet(model=model, epsilon=epsilon)
        else:
            raise ValueError("Not implemented yet.")

        model.summary()

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        if dataset_type == 4:
            """Need to impose some maximum value on clipnorm to be stable for larger networks"""
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=0.5)
        loss = Crossentropy(sparse=True)
        metrics = ["clean_acc", "vra"]

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        save_callback = GloroSaveCallback(path=model_save_path + "/model.hdf5")

        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size,
                callbacks=[UpdatePowerIterates(), save_callback])

