import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import foolbox as fb
from foolbox.criteria import TargetedMisclassification

from models import model_AT_mnist, model_AT_cifar, model_AT_mnist_deelLip, model_AT_cifar_deelLip, res_net9, celeba_model
from utils import load_dataset
from plot_TLA_embeddings import manual_pgd_attack
from AT_utils import make_adv_batch


datasets = {
    1: "MNIST",
    2: "FMNIST",
    3: "CIFAR-10",
    4: "eurosat",
    5: "CelebA"
}


mnist_paths = [
    "models_AT/simple/fgsm/MNIST/model_epsilon=0.5.hdf5",
    "models_TRADES/MNIST/delta=2/epsilon=1.0/model.hdf5",
    "models_TLA/MNIST/lambd=0.5/margin=0.05/model.hdf5",
    "models_DeepDefense/MNIST/lambda/factor=5_c=0.05_exp/model.hdf5",
    "models_MACER/MNIST/sigma=0.1/lambd=12/model.hdf5",
    "models_GloroNet/MNIST/epsilon=0.6/model.hdf5",
    ("models_DeelLip/MNIST/per_layer_lip=1/model.hdf5", 1)
]

fmnist_paths = [
    "models_AT/simple/fgsm/FMNIST/model_epsilon=0.5.hdf5",
    "models_TRADES/FMNIST/delta=1/epsilon=1.0/model.hdf5",
    "models_TLA/FMNIST/lambd=4/margin=0.05/model.hdf5",
    "models_DeepDefense/FMNIST/lambda/factor=1_c=0.01_exp/model.hdf5",
    "",
    "models_GloroNet/FMNIST/epsilon=0.5/model.hdf5",
    "models_DeelLip/FMNIST/per_layer_lip=0.75_withClip/model.hdf5"
]
    
cifar_paths = [
    "models_AT/simple/fgsm/CIFAR-10/model_epsilon=0.3.hdf5",
    "models_TRADES/CIFAR-10/delta=6/epsilon=0.5/model.hdf5",
    "models_TLA/CIFAR-10/lambd=4/margin=0.05/model.hdf5",
    "models_DeepDefense/CIFAR-10/lambda/factor=1_c=0.01_exp/model.hdf5",
    "models_MACER/CIFAR-10/sigma=0.3/lambd=0.01/model.hdf5",
    "models_GloroNet/CIFAR-10/epsilon=0.05/model.hdf5",
    "models_DeelLip/CIFAR-10/per_layer_lip=0.75_withClip_freeLip/model.hdf5"
]

eurosat_paths = [
    "models_AT/simple/fgsm/eurosat/model_epsilon=0.1.hdf5",
    "models_TRADES/eurosat/delta=3/epsilon=0.3/model.hdf5",
    "models_TLA/eurosat/lambd=1/margin=0.1/model.hdf5",
    "models_DeepDefense/eurosat/lambda/factor=1_c=0.01_exp/model.hdf5",
    "models_MACER/eurosat/sigma=0.1/lambd=0.1/model.hdf5",
    "models_GloroNet/eurosat/epsilon=500.0/model.hdf5",
    "models_DeelLip/eurosat/per_layer_lip=0.5_withClip/model.hdf5",
]

celeba_paths = [
    "models_AT/simple/fgsm/CelebA/model_epsilon=0.1.hdf5",
    "models_TRADES/CelebA/delta=50/epsilon=0.3/model.hdf5",
    "models_TLA/CelebA/lambd=0.5/margin=0.1/model.hdf5",
    "models_DeepDefense/CelebA/lambda/factor=0.1_c=0.01_exp/model.hdf5",
    "models_MACER/CelebA/sigma=0.1/lambd=0.1/model.hdf5",
    "models_GloroNet/CelebA/epsilon=0.3/model.hdf5",
    "models_DeelLip/CelebA/per_layer_lip=0.55_withClip_freeLip/model.hdf5",
]

names = ["AT", "TRADES", "TLA", "DeepDefense", "MACER", "GloroNet", "DeelLip (SVD)"] # pre-defined order for all datasets
datasets_paths = [mnist_paths, fmnist_paths, cifar_paths, eurosat_paths, celeba_paths]


if __name__ == "__main__":

    dataset_type = 1 # also used to select the list of paths to load
    idx_methods = [0, 1, 2, 3, 4, 5, 6] # used to select which methods from names list will be plotted 
    n_samples = 10 # number of figures to generate & save
    class_id = 3 # if not None, attack n_samples from this class
    target_id = 8 # if not None, samples from class class_id are attacked s.t. the adversarial sample should have class target_id
    head = True
    plot_text = True

    paths = datasets_paths[dataset_type - 1]
    paths = [paths[idx] for idx in idx_methods]
    save_path = f"adv_examples/{datasets[dataset_type]}/source={class_id}_target={target_id}/"

    _, _, x_test, y_test, input_shape, n_classes = load_dataset(dataset_type)

    models = []
    names = ["Baseline"] + [names[idx] for idx in idx_methods]

    if dataset_type in [1, 2]:
        n_steps = 50
        step_size = 0.025
    elif dataset_type == 3:
        n_steps = 50
        step_size = 0.01
    elif dataset_type in [4, 5]:
        n_steps = 50
        step_size = 0.005

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

    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    idx_test = np.array(list(range(x_test.shape[0])))
    if class_id is not None:
        idx_test = idx_test[y_test == class_id]

    idx_test_attack = np.random.choice(idx_test, size=n_samples, replace=False)

    x_test_batch = x_test[idx_test_attack]
    models_adv = {}

    for i, m in enumerate(names):
        x_test_adv = np.empty_like(x_test_batch)
        if target_id is not None:
            # for k in range(x_test_batch.shape[0]):
            #     x_test_adv[k] = manual_pgd_attack(x_test_batch[k][np.newaxis, ...], [target_id], models[i], n_steps=n_steps, step_size=step_size, 
            #                                       targeted=True, exit_if_crossed=True)
            fmodel = fb.models.TensorFlowModel(models[i], bounds=(-1, 1))
            attack = fb.attacks.L2DeepFoolAttack(steps=100)
            _, x_test_adv, _ = attack(fmodel, 
                                    tf.cast(x_test_batch, float),
                                    epsilons=100, criterion=TargetedMisclassification(target_classes=tf.cast([target_id] * x_test_batch.shape[0], tf.int32)))
            x_test_adv = x_test_adv.numpy()
        else:
            fmodel = fb.models.TensorFlowModel(models[i], bounds=(-1, 1))
            attack = fb.attacks.L2DeepFoolAttack(steps=100)
            _, x_test_adv, _ = make_adv_batch(fmodel, attack, 1000, 
                                        tf.cast(x_test_batch, float),
                                        tf.cast([class_id] * x_test_batch.shape[0], tf.int32),
                                        batch_size=n_samples
                                        )

        models_adv[m] = x_test_adv * 127.5 + 127.5

    for i in range(x_test_batch.shape[0]):
        
        x_test_batch[i] = x_test_batch[i] * 127.5 + 127.5

        figsize = (10, 4)
        fig = plt.figure(figsize=figsize)

        plt.subplot(1, len(models) + 1, 1)
        plt.imshow(x_test_batch[i].astype(np.uint8), cmap='gray')
        if head:
            plt.title("Original", fontsize=12)
        plt.yticks([])
        plt.xticks([])

        for j, m in enumerate(names):
            plt.subplot(1, len(models) + 1, j + 2)
            plt.imshow(models_adv[m][i].astype(np.uint8), cmap='gray')
            if head:
                plt.title(names[j], fontsize=12)
            plt.yticks([])
            plt.xticks([])

            diff = np.abs(models_adv[m][i].astype(float) - x_test_batch[i].astype(float))
            if plot_text:
                y = 15 if dataset_type in [4, 5] else 4
                plt.text(0, y, "{:.2f}".format(np.linalg.norm(diff.flatten() / 255, ord=2)), color="r", fontsize=10)

        fig.tight_layout()
        plt.savefig(F"{save_path}/sample_{idx_test_attack[i]}.png", bbox_inches="tight", dpi=300, pad_inches=0.01,
                        orientation='landscape', transparent=True)



