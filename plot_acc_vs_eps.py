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


datasets = {
    1: "MNIST",
    2: "FMNIST",
    3: "CIFAR-10",
    4: "eurosat",
    5: "CelebA"
}


def attack_model_foolbox(model, x, y, eps, attack_name='ddn'):
    fmodel = fb.models.TensorFlowModel(model, bounds=(-1, 1))
    x = tf.cast(x, float)
    y = tf.cast(y, tf.int32)

    if attack_name == 'ddn':
       attack = fb.attacks.DDNAttack(steps=300)
    elif attack_name == "deepfool":
       attack = fb.attacks.L2DeepFoolAttack(steps=300)
    elif attack_name == "fmn":
       attack = fb.attacks.L2FMNAttack(steps=300)
    else:
       raise ValueError(F"Attack {attack_name} not implemented.")

    raw, clipped, is_adv = make_adv_batch(fmodel, attack, eps, x, y, batch_size=1024)

    return clipped, tf.cast(is_adv, float)


if __name__ == "__main__":
   
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    attack_name = "ddn"
    attack_first_n = 1000
    dataset_type = 3
    
    save_plot = True
    name = "comparison_sigma"
    method = "MACER"
    save_path = f"acc_eps_plots/{datasets[dataset_type]}/{method}/attack={attack_name}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    paths = [
        "models_MACER/CIFAR-10/sigma=0.1/lambd=0.1/model.hdf5",
        "models_MACER/CIFAR-10/sigma=0.2/lambd=0.1/model.hdf5",
        "models_MACER/CIFAR-10/sigma=0.3/lambd=0.1/model.hdf5",
        "models_MACER/CIFAR-10/sigma=0.4/lambd=0.1/model.hdf5",
    ]
    model_names = [
        "$\lambda = 0.1 \, \sigma=0.1$",
        "$\lambda = 0.1 \, \sigma=0.2$",
        "$\lambda = 0.1 \, \sigma=0.3$",
        "$\lambda = 0.1 \, \sigma=0.4$",
    ]

    # # DeelLip models should be provided as tuple (file_path, lip_constant)
    # paths = [
    #     "models_TRADES/CIFAR-10/delta=6/epsilon=0.5/model.hdf5",
    #     "models_TRADES/CIFAR-10/bhatt/delta=6_epsilon=0.5/model.hdf5",
    #     "models_TRADES/CIFAR-10/emd/delta=0.01_epsilon=0.5/model.hdf5",
    #     "models_TRADES/CIFAR-10/jsd/delta=6_epsilon=0.5/model.hdf5"
    # ]
    # # model_names will contain the name to be plotted along acc vs eps 
    # model_names = [
    #     "Kullback-Leibler", "Bhattacharya", "Earth Mover's Distance", "Jensen-Shannon"
    # ]

    models = []

    x_train, y_train, x_test, y_test, input_shape, n_classes = load_dataset(dataset_type)

    x_test = x_test[:attack_first_n]
    y_test = y_test[:attack_first_n]

    if dataset_type == 1:
        eps_values = [0.8 * i for i in range(11)]
    elif dataset_type == 2:
        eps_values = [0.8 * i for i in range(11)]
    elif dataset_type == 3:
        eps_values = [0.2 * i for i in range(11)]
    elif dataset_type == 4:
        eps_values = [0.25 * i for i in range(11)]
        attack_first_n = 500
    elif dataset_type == 5:
        eps_values = [1.0 * i for i in range(11)]

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
    else:
        raise ValueError("Not implemented yet.")
    models.append(model_base)
    model_names.append("baseline")

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    for i, model in enumerate(models):
        accs = []

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ['accuracy']
        model.trainable = False
        model.compile(optimizer, loss, metrics)
        y_pred = model(x_test)
        model.evaluate(x_test, y_test)
        
        model.trainable = False
        clipped, is_adv = attack_model_foolbox(model, x_test, y_test, eps_values, attack_name=attack_name)
        robust_accuracy = 1 - is_adv.numpy().mean(axis=-1)
        print(robust_accuracy)
        if save_plot:
            plt.plot(np.array(eps_values) / 2, robust_accuracy * 100, label=model_names[i])
    if save_plot:
        plt.legend()
        plt.grid()
        plt.ylabel("Acc [%]")
        plt.xlabel("Max. $||\eta||_2$")
        ax.set_yticks([i * 10 for i in range(11)])
        plt.ylim([0, 100])
        plt.xlim([0, np.max(eps_values) / 2])
        plt.savefig(save_path + "/" + name + ".png", bbox_inches="tight", dpi=300, pad_inches=0.01, orientation='landscape', transparent=True)

     
   