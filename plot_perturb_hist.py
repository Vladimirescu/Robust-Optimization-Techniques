import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
import numpy as np
import foolbox as fb
from models import model_AT_mnist, model_AT_mnist_deelLip, model_AT_cifar, model_AT_cifar_deelLip, res_net9, celeba_model
from AT_utils import make_adv_batch
from gloro import GloroNet
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import load_dataset

from plot_acc_vs_eps import attack_model_foolbox


datasets = {
    1: "MNIST",
    2: "FMNIST",
    3: "CIFAR-10",
    4: "eurosat",
    5: "CelebA",
    6: "BelgiumTS",
}


if __name__ == "__main__":

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    attack_name = "ddn"
    attack_first_n = 1000
    dataset_type = 5
    
    name = "hist_perturb_distances"
    method = "all"
    save_path = f"acc_eps_plots/{datasets[dataset_type]}/{method}/attack={attack_name}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # # DeelLip models should be provided as tuple (file_path, lip_constant)
    paths = [
        # "models_AT/simple/fgsm/CelebA/model_epsilon=0.1.hdf5",
        # "models_TRADES/CelebA/delta=50/epsilon=0.3/model.hdf5",
        "models_TLA/CelebA/lambd=0.5/margin=0.1/model.hdf5",
        # "models_DeepDefense/CelebA/lambda/factor=0.1_c=0.01_exp/model.hdf5",
        # "models_MACER/CelebA/sigma=0.1/lambd=0.1/model.hdf5",
        # "models_GloroNet/CelebA/epsilon=0.3/model.hdf5",
        # "models_DeelLip/CelebA/per_layer_lip=0.55_withClip_freeLip/model.hdf5",
        "models_LMT/CelebA/c=25/model.hdf5"
    ]
    # model_names will contain the name to be plotted along acc vs eps 
    model_names = [
        "TLA", "LMT"
    ]

    models = []

    x_train, y_train, x_test, y_test, input_shape, n_classes = load_dataset(dataset_type)

    x_test = x_test[:attack_first_n]
    y_test = y_test[:attack_first_n]

    "Load constrained models"
    for i, path in enumerate(paths):

        if isinstance(path, tuple):
            """For deel models, the path is given as (path_file, k_coef_lip)"""
            path, lip = path
            deel = True
        else:
            deel = False

        if dataset_type in [1, 2]:
            if deel:
                model = model_AT_mnist_deelLip(input_shape, n_classes, k_coef=lip)
            else:
                model = model_AT_mnist(input_shape, n_classes, batch_norm=False)
        elif dataset_type == 3:
            if deel:
                model = model_AT_cifar_deelLip(input_shape, n_classes, k_coef=lip)
            else:
                model = model_AT_cifar(input_shape, n_classes, batch_norm=False)
        elif dataset_type == 4:
            model = res_net9(input_shape, n_classes, deel=True)
        elif dataset_type == 5:
            model = celeba_model(input_shape, n_classes)
        elif dataset_type == 6:
            model = res_net9(input_shape, n_classes, deel=True, filters_start=8)
        else:
            raise ValueError("Not implemented yet.")
        model.load_weights(path)
        
        models.append(model)

        """Load baseline"""
    if dataset_type in [1, 2]:
        model_base = model_AT_mnist(input_shape, n_classes, batch_norm=False)
        if dataset_type == 1:
            model_base.load_weights("models_AT/baseline/MNIST/model.hdf5")
        else:
            model_base.load_weights("models_AT/baseline/FMNIST/model.hdf5")
    elif dataset_type == 3:
        model_base = model_AT_cifar(input_shape, n_classes, batch_norm=False)
        model_base.load_weights("models_AT/baseline/CIFAR-10/model.hdf5")
    elif dataset_type == 4:
        model_base = res_net9(input_shape, n_classes)
        model_base.load_weights("models_AT/baseline/eurosat/model.hdf5")
    elif dataset_type == 5:
        model_base = celeba_model(input_shape, n_classes)
        model_base.load_weights("models_AT/baseline/CelebA/model.hdf5")
    elif dataset_type == 6:
        model_base = res_net9(input_shape, n_classes, deel=True, filters_start=8)
        model_base.load_weights("models_AT/baseline/BelgiumTS/model.hdf5")
    else:
        raise ValueError("Not implemented yet.")
    models.append(model_base)
    model_names.append("baseline")

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    max_perturb = 0
    for i, model in enumerate(models):
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ['accuracy']
        model.trainable = False
        model.compile(optimizer, loss, metrics)

        y_pred = tf.math.argmax(model(x_test), -1).numpy()

        clipped, is_adv = attack_model_foolbox(model, x_test, y_test, None, attack_name=attack_name)

        example_mask = (y_pred == y_test) * (is_adv.numpy() == True)

        clipped = clipped[example_mask]
        clipped = clipped.reshape((clipped.shape[0], -1))
        x_original = x_test[example_mask]
        x_original = x_original.reshape((x_original.shape[0], -1))

        perturbs = np.linalg.norm((clipped - x_original), ord=2, axis=-1)
        q99 = np.quantile(perturbs, 0.99)
        max_perturb = q99 if q99 > max_perturb else max_perturb

        plt.hist(perturbs, bins=150, label=model_names[i], density=True, alpha=1.5/len(models))
  
    plt.legend()
    plt.xlabel("$||\eta||_2$")
    ax.set_yticks([])
    plt.xlim([0, max_perturb / 2])
    ax.set_xticks(np.arange(0, max_perturb / 2, 2))
    plt.savefig(save_path + "/" + name + ".png", bbox_inches="tight", dpi=300, pad_inches=0.01, orientation='landscape', transparent=True)
