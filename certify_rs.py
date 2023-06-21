import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from art.estimators.certification.randomized_smoothing import TensorFlowV2RandomizedSmoothing
from models import model_AT_mnist, model_AT_cifar

from utils import load_dataset


datasets = {
    1: "MNIST",
    2: "FMNIST",
    3: "CIFAR-10"
}


if __name__ == "__main__":
    
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    dataset_type = 3
    test_first_n = 1000

    save_plot = True
    name = "comparison"
    method = "RS_MACER"
    save_path = f"rs_plots/{datasets[dataset_type]}/{method}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    paths = [
        "models_RS/CIFAR-10/sigma=0.1/k=2/model.hdf5",
        "models_RS/CIFAR-10/sigma=0.2/k=2/model.hdf5",
        "models_MACER/CIFAR-10/sigma=0.1/lambd=0.05/model.hdf5",
        "models_MACER/CIFAR-10/sigma=0.2/lambd=0.05/model.hdf5"
    ]
    # model_names will contain the name to be plotted along acc vs eps 
    model_names = [
        "$RS \, \sigma = 0.2$",    
        "$RS \, \sigma = 0.3$",
        "$MACER \, \sigma = 0.2$",
        "$MACER \, \sigma = 0.3$",
    ]

    models = []

    x_train, y_train, x_test, y_test, input_shape, n_classes = load_dataset(dataset_type)

    "Load constrained models"
    for i, path in enumerate(paths):
        if dataset_type in [1, 2]:
            model = model_AT_mnist(input_shape, n_classes, batch_norm=False)

            if dataset_type == 1:
                scales = [0.5 * i for i in range(11)]
            else:
                scales = [0.25 * i for i in range(11)]
        elif dataset_type == 3:
            model = model_AT_cifar(input_shape, n_classes, batch_norm=False)

            scales = [0.2 * i for i in range(11)]
        else:
            raise ValueError("Not implemented yet.")
        model.load_weights(path)
        
        models.append(model)

    if dataset_type in [1, 2]:
        model_base = model_AT_mnist(input_shape, n_classes, batch_norm=False)
        if dataset_type == 1:
            model_base.load_weights("models_AT/baseline/MNIST/model.hdf5")
        else:
            model_base.load_weights("models_AT/baseline/FMNIST/model.hdf5")
    elif dataset_type == 3:
        model_base = model_AT_cifar(input_shape, n_classes, batch_norm=False)
        model_base.load_weights("models_AT/baseline/CIFAR-10/model.hdf5")
    models.append(model_base)
    model_names.append("baseline")

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    max_r = 0
    max_last_acc = 0
    for i, model in enumerate(models):

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ['accuracy']
        model.trainable = False
        model.compile(optimizer, loss, metrics)

        accs = []
        max_rs = []
        max_r_scale = 0
        for scale in scales:
            certify = TensorFlowV2RandomizedSmoothing(
                model, 
                nb_classes=n_classes,
                input_shape=input_shape,
                channels_first=False,
                clip_values=(-1, 1),
                scale=scale
            )        

            c, r = certify.certify(x_test[:test_first_n], n=100, batch_size=256)
            acc = np.mean(c == y_test[:test_first_n])
            
            accs.append(acc)
            """Get the average radius, between al radius != 0"""
            try:
                mx_r = np.max(r[r != 0])
            except:
                mx_r = 0
            max_r_scale = max_r_scale if mx_r < max_r_scale else mx_r
            max_rs.append(max_r_scale)

        print(accs)
        print(max_rs)

        max_r = np.max(max_rs) if np.max(max_rs) > max_r else max_r
        max_last_acc = accs[-1] if accs[-1] > max_last_acc else max_last_acc

        if save_plot:
            plt.plot(max_rs, np.array(accs) * 100, label=model_names[i])
    if save_plot:
        plt.legend()
        plt.grid()
        plt.ylabel("Acc [%]")
        plt.xlabel("Max. $R$")
        plt.ylim([max_last_acc, 100])
        ax.set_yticks([i * 10 for i in range(int(max_last_acc) + 1, 11)])
        plt.xlim([0, int(max_r) + 1])
        ax.set_xticks(np.arange(0, int(max_r) + 1, 0.5))
        plt.savefig(save_path + "/" + name + ".png", bbox_inches="tight", dpi=300, pad_inches=0.01, orientation='landscape', transparent=True)