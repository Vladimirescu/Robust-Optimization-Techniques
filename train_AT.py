import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
import numpy as np
import os
from models import model_AT_mnist, wide_resnet, model_AT_cifar, res_net9, celeba_model
from AT_utils import *
from utils import load_dataset


datasets = {
    1: "MNIST",
    2: "FMNIST",
    3: "CIFAR-10",
    4: "eurosat",
    5: "CelebA"
}


at_types = {
    "baseline": {"epochs": 50},
    "simple": {
        "epochs": 50,
        "batch_size": 512
    },
    "ensemble": {
        "epochs": 30,
        "n_ensembles": 5,
        "epochs_ensemble": 30,
        "batch_size": 512
    },
}


if __name__ == "__main__":
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for i in range(len(gpus)):
            tf.config.experimental.set_memory_growth(gpus[i], True)

    """Settings"""
    dataset_type = 5
    filename = "model"
    base_folder = "models_AT/"
    at_type = "simple" # adversarial example algorithm type
    attack_used = "fgsm" # attack used for generating adversarial samples during traing
    log_metrics = False
    """---"""

    assert dataset_type in datasets.keys(), ValueError(f"Dataset type {dataset_type} not implemented.")
    assert at_type in at_types.keys(), ValueError(f"Algorithm {at_type} not implemented.")
    assert attack_used in at_attack_types.keys(), ValueError(f"Avdresarial training for attack {attack_used} not immplemented.")

    dset_name = datasets[dataset_type]

    x_train, y_train, x_test, y_test, input_shape, n_classes = load_dataset(dataset_type)

    if not os.path.exists(base_folder + at_type):
        os.mkdir(base_folder + at_type)

    if at_type == "baseline":
        model_save_path = base_folder + at_type + "/" + datasets[dataset_type] 
        os.makedirs(model_save_path, exist_ok=True)
    elif at_type == "simple":
        model_save_path = base_folder + at_type + "/" + attack_used + "/" + datasets[dataset_type]
        os.makedirs(model_save_path, exist_ok=True)
    elif at_type == "ensemble":
        model_save_path = base_folder + at_type + "/" + "ensembles=" + str(at_types[at_type]["n_ensembles"]) + "/" + attack_used + "/" + datasets[dataset_type]
        os.makedirs(model_save_path, exist_ok=True)

    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['accuracy']

    if dataset_type in [1, 2]:
        model = model_AT_mnist(input_shape=input_shape, n_classes=n_classes, batch_norm=False)
    elif dataset_type == 3:
        model = model_AT_cifar(input_shape=input_shape, n_classes=n_classes, batch_norm=False)
        model.summary()
    elif dataset_type == 4:
        model = res_net9(input_shape=input_shape, n_classes=n_classes, deel=True)
        model.summary()
    elif dataset_type == 5:
        model = celeba_model(input_shape=input_shape, n_classes=n_classes)
        model.summary()

    model.compile(optimizer, loss, metrics)

    if at_type == "baseline":
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path + "/" + filename + ".hdf5", monitor='val_loss',
                                                    verbose=1, save_best_only=True, save_weights_only=True)

        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=at_types[at_type]["epochs"], batch_size=64,
                  callbacks=[
                    checkpoint
                  ])
    elif at_type == "simple":
        adversarial_training_simple(
            dset_name, model, x_train, y_train, x_test, y_test, epochs=at_types[at_type]["epochs"],
            batch_size=at_types[at_type]["batch_size"], attack=attack_used, save_names=(model_save_path, filename), log_metrics=log_metrics
        )
    elif at_type == "ensemble":
        ensemble_adversarial_training(
            dset_name, model, x_train, y_train, x_test, y_test, epochs=at_types[at_type]["epochs"], epochs_ensemble=at_types[at_type]["epochs_ensemble"],
            batch_size=at_types[at_type]["batch_size"], n_ensembles=at_types[at_type]["n_ensembles"], 
            attack=attack_used, save_names=(model_save_path, filename), log_metrics=log_metrics
        )