import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
import numpy as np
import os
from models import model_AT_mnist, model_AT_cifar, res_net9


datasets = {
    1: "MNIST",
    2: "FMNIST",
    3: "CIFAR-10",
    4: "eurosat"
}


if __name__ == "__main__":

    ###
    batch_size = 128
    dataset_type = 4
    ###

    dset_name = datasets[dataset_type]
    if dataset_type == 1:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = (x_train - 127.5) / 127.5
        x_test = (x_test - 127.5) / 127.5
        input_shape = (28, 28, 1)
        n_classes = 10
    elif dataset_type == 2:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = (x_train - 127.5) / 127.5
        x_test = (x_test - 127.5) / 127.5
        input_shape = (28, 28, 1)
        n_classes = 10
    elif dataset_type == 3:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        x_train = (x_train - 127.5) / 127.5
        x_test = (x_test - 127.5) / 127.5
        input_shape = (32, 32, 3)
        n_classes = 10
    elif dataset_type == 4:
        k = 2

        ds, ds_info = tfds.load('eurosat/rgb',
                            with_info=True,
                            split='train',
                            data_dir="/scratch/")
        
        X = []
        y = []
        for i, x in enumerate(ds):
            X.append(x["image"].numpy().astype(float))
            y.append(x["label"].numpy().astype(int))

        X = (np.array(X) - 127.5) / 127.5
        y = np.array(y)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

        input_shape = (64, 64, 3)
        n_classes = 10
        epochs = 200
        batch_size = 64

    for sigma in [0.1, 0.2]:
        for k in [k]:
            model_save_path = f"models_RS/{datasets[dataset_type]}/sigma={sigma}/k={k}"
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            if dataset_type in [1, 2]:
                epochs = 20
                model = model_AT_mnist(input_shape=input_shape, n_classes=n_classes, batch_norm=False)
            elif dataset_type == 3:
                epochs = 30
                model = model_AT_cifar(input_shape=input_shape, n_classes=n_classes, batch_norm=False)
            elif dataset_type == 4:
                epochs = 100
                model = res_net9(input_shape=input_shape, n_classes=n_classes, deel=True)
            else:
                raise ValueError("Not implemented yet.")
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
            if dataset_type == 4:
                optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metrics = ['accuracy']
            model.compile(optimizer, loss, metrics)
            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path + "/model.hdf5", monitor='val_loss',
                                                    verbose=1, save_best_only=True, save_weights_only=True)
            
            for e in range(epochs):
                """Construct k noisy examples for each image from x_train, at each epoch"""
                x_train_n = np.repeat(x_train, k, axis=0)
                x_train_n = x_train_n + np.random.normal(0, sigma, x_train_n.shape)
                y_train_n = np.repeat(y_train, k, axis=0)

                model.fit(x_train_n, y_train_n, validation_data=(x_test, y_test), epochs=1, batch_size=batch_size,
                        callbacks=[checkpoint])
        
