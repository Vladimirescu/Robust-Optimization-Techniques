import tensorflow as tf
import foolbox as fb
import numpy as np
from models import model_AT_mnist, model_AT_cifar
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def manual_pgd_attack(x_batch, target_batch, model, n_steps, step_size, targeted=True, exit_if_crossed=False):
    """
    Targeted PGD attack
    
    Given examples x_batch, assumed to be of class != target_batch,
    this functions aims at finding those adversarial samples that would
    minimize the loss associated with classifying them as class target_batch.

    If targeted = True, the above behaviour is used.
    Else, targeted represents the correct class, and the direction corresponds to the loss maximization w.r.t. it.

    exit_if_corssed: only to be used when working with 1 image at the time - if True, exits the loop when the target has been attained
    """
    x_batch_adv = tf.cast(x_batch, float)
    loss_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for i in range(n_steps):
        with tf.GradientTape() as tape_x:
            tape_x.watch(x_batch_adv)
            logits = model(x_batch_adv)
            loss = loss_cls(target_batch, logits)

        gradient = tape_x.gradient(loss, x_batch_adv)
        sign_batch = tf.sign(gradient)

        if targeted:
            x_batch_adv = tf.clip_by_value(x_batch_adv - step_size * sign_batch, -1 , 1)
        else:
            x_batch_adv = tf.clip_by_value(x_batch_adv + step_size * sign_batch, -1 , 1)

        if exit_if_crossed and x_batch_adv.shape[0] == 1:
            class_pred = tf.math.argmax(model(x_batch_adv), axis=-1)
            
            if targeted and class_pred == target_batch:
                print(f"Exited in {i+1} iterations.")
                return x_batch_adv
            if not targeted and class_pred != target_batch:
                print(f"Exited in {i+1} iterations.")
                return x_batch_adv

    return x_batch_adv


datasets = {
    1: "MNIST",
    2: "FMNIST",
    3: "CIFAR-10"
}


if __name__ == "__main__":
    ###
    dataset_type = 1

    path = "models_TLA/MNIST/lambd=0.5/margin=0.1/model.hdf5"
    name = "mnist_0.5_0.1"

    save_path = "TLA_plots/" + name + ".png"
    ###

    dset_name = datasets[dataset_type]
    if dataset_type == 1:

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = (x_train - 127.5) / 127.5
        x_test = (x_test - 127.5) / 127.5
        input_shape = (28, 28, 1)
        n_classes = 10

        class_names = [str(i) for i in range(10)]
        attack_steps = 20
        step_size = 0.1
        class_positive = 0
        class_negative = 6
    elif dataset_type == 2:

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = (x_train - 127.5) / 127.5
        x_test = (x_test - 127.5) / 127.5
        input_shape = (28, 28, 1)
        n_classes = 10

        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        attack_steps = 20
        step_size = 0.2
        class_positive = 4
        class_negative = 7

    elif dataset_type == 3:

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        x_train = (x_train - 127.5) / 127.5
        x_test = (x_test - 127.5) / 127.5
        input_shape = (32, 32, 3)
        n_classes = 10

        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 
                       'frog', 'horse', 'ship', 'truck']
        attack_steps = 5
        step_size = 0.025
        class_positive = 0
        class_negative = 5

    if dataset_type in [1, 2]:
        model = model_AT_mnist(input_shape=input_shape, n_classes=n_classes, batch_norm=False)
        model_embedd = model_AT_mnist(input_shape=input_shape, n_classes=n_classes, batch_norm=False, return_embeddings=True)
    elif dataset_type == 3:
        model = model_AT_cifar(input_shape=input_shape, n_classes=n_classes, batch_norm=False)
        model_embedd = model_AT_cifar(input_shape=input_shape, n_classes=n_classes, batch_norm=False, return_embeddings=True)
    else:
        raise ValueError("Dataset unknown.")
    
    model.load_weights(path)
    model_embedd.load_weights(path)

    '''Sample positives/negatives'''
    np.random.seed(33)

    pred = model.predict(x_test, batch_size=512, verbose=0)
    pred = np.argmax(pred, axis=-1)

    pos_mask = (y_test == class_positive) * (pred == y_test)
    neg_mask = (y_test == class_negative) * (pred == y_test)

    pos_samples = x_test[pos_mask]
    neg_samples = x_test[neg_mask]

    samples_per_class = min([pos_samples.shape[0], neg_samples.shape[0]])

    idx_pos = np.random.choice(list(range(len(pos_samples))), size=samples_per_class, replace=False)
    idx_neg = np.random.choice(list(range(len(neg_samples))), size=samples_per_class, replace=False)

    pos_samples = pos_samples[idx_pos, ...]
    neg_samples = neg_samples[idx_neg, ...]

    """Make adv samples"""
    target_class = [class_negative for _ in range(pos_samples.shape[0])]
    adv_samples = manual_pgd_attack(pos_samples, target_class, model, n_steps=attack_steps, step_size=step_size)

    """Get embeddings"""
    _, embeddings_pos = model_embedd(pos_samples)
    _, embeddings_adv = model_embedd(adv_samples)
    _, embeddings_neg = model_embedd(neg_samples)

    embeddings_all = np.concatenate((embeddings_pos, embeddings_adv, embeddings_neg), axis=0)
    """Project on 2D space"""
    tsne = TSNE(n_components=2, perplexity=40, metric='cosine', random_state=33)
    embeddings_all_2d = tsne.fit_transform(embeddings_all)

    pos_2d, adv_2d, neg_2d = np.split(embeddings_all_2d, 3, axis=0)

    """Plot points"""
    plt.figure(figsize=(5, 5))
    plt.scatter(pos_2d[:, 0], pos_2d[:, 1], s=40, c="g", marker="+", label=f"{class_names[class_positive]} (positive)", alpha=0.3)
    plt.scatter(adv_2d[:, 0], adv_2d[:, 1], s=40, c="r", marker="x", label="adversarial", alpha=0.3)
    plt.scatter(neg_2d[:, 0], neg_2d[:, 1], s=40, c="b", marker="o", label=f"{class_names[class_negative]} (negative)", alpha=0.3)
    plt.legend(loc="best", prop={'size': 12})
    # plt.axis("off")
    plt.tick_params(axis='both', direction='in', labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    plt.savefig(save_path, pad_inches=0.01, dpi=300, bbox_inches="tight")


