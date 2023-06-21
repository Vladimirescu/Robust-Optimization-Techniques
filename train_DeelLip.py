import tensorflow as tf
import numpy as np
import os
from models import model_AT_mnist_deelLip, model_AT_cifar_deelLip, model_AT_mnist, model_AT_cifar, res_net9, celeba_model
from utils import SingularValueClip
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from utils import load_dataset


datasets = {
    1: "MNIST",
    2: "FMNIST",
    3: "CIFAR-10",
    4: "eurosat",
    5: "CelebA"
}


if __name__ == "__main__":

    ###
    batch_size = 512
    dataset_type = 5
    with_clip = True
    free_lips = True
    ###

    dset_name = datasets[dataset_type]

    x_train, y_train, x_test, y_test, input_shape, n_classes = load_dataset(dataset_type)

    if dataset_type in [1, 2]:
        epochs = 50
    elif dataset_type == 3:
        epochs = 100
    elif dataset_type in [4, 5]:
        epochs = 200

    for per_layer_lip in [0.55]:
        if with_clip:
            if free_lips:
                model_save_path = f"models_DeelLip/{datasets[dataset_type]}/per_layer_lip={per_layer_lip}_withClip_freeLip/"
            else:
                model_save_path = f"models_DeelLip/{datasets[dataset_type]}/per_layer_lip={per_layer_lip}_withClip/"
        else:
            model_save_path = f"models_DeelLip/{datasets[dataset_type]}/per_layer_lip={per_layer_lip}/"
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if dataset_type in [1, 2]:
            if with_clip:
                model = model_AT_mnist(input_shape=input_shape, n_classes=n_classes, batch_norm=False)
            else:
                model = model_AT_mnist_deelLip(input_shape=input_shape, n_classes=n_classes, k_coef=per_layer_lip)
        elif dataset_type == 3:
            if with_clip:
                model = model_AT_cifar(input_shape=input_shape, n_classes=n_classes, batch_norm=False)
                model.load_weights("models_DeelLip/CIFAR-10/per_layer_lip=0.75_withClip/model.hdf5")
            else:
                model = model_AT_cifar_deelLip(input_shape=input_shape, n_classes=n_classes, k_coef=per_layer_lip)
        elif dataset_type == 4:
            if with_clip:
                model = res_net9(input_shape, n_classes, deel=True)
                model.load_weights("models_DeelLip/eurosat/per_layer_lip=0.5_withClip/model.hdf5")
                batch_size = 128
            else:
                pass
        elif dataset_type == 5:
            if with_clip:
                model = celeba_model(input_shape, n_classes)
                model.load_weights("models_DeelLip/CelebA/per_layer_lip=0.6_withClip_freeLip/model.hdf5")
                batch_size = 128
            else:
                pass
        else:
            raise ValueError("Not implemented yet.")
        
        model.summary()

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        if dataset_type in [4, 5]:
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=0.5)

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ['accuracy']
        model.compile(optimizer, loss, metrics)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path + "/model.hdf5", monitor='val_loss',
                                                verbose=1, save_best_only=True, save_weights_only=True)
        if with_clip:
            svd_constraint = SingularValueClip(lip=per_layer_lip, free_lips=free_lips)

            model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size,
                      callbacks=[svd_constraint, checkpoint])
        else:
            model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size,
                    callbacks=[checkpoint])
    
