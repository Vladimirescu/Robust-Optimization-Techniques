import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from models import model_AT_mnist, model_AT_cifar, res_net9, celeba_model
from AT_utils import make_adv_batch
import foolbox as fb
import numpy as np
from tqdm import tqdm
from utils import load_dataset


datasets = {
    1: "MNIST",
    2: "FMNIST",
    3: "CIFAR-10",
    4: "eurosat",
    5: "CelebA"
}


def jensen_shannon(x, y):
    """
    Computes the Jensen-Shannon Divergence between discrete probability distributions x and y.
    """
    m = (x + y) / 2.0

    kl_1 = tf.keras.losses.KLDivergence(reduction='none')(x, m)
    kl_2 = tf.keras.losses.KLDivergence(reduction='none')(y, m)

    return tf.math.reduce_mean((kl_1 + kl_2) / 2.0)


def earth_mover_distance(x, y):
    """
    Computes the Earth Mover Distance between discrete probability distributions x and y.
    """

    # compute CDF for each sample
    X = tf.math.cumsum(x, axis=-1)
    Y = tf.math.cumsum(y, axis=-1)

    n = x.shape[-1]

    # make n x n matrix of values |i - j|
    factors = tf.abs(tf.subtract(tf.expand_dims(tf.range(n), 0), tf.expand_dims(tf.range(n), 1)))
    factors = tf.cast(factors, float)

    # compute CDF differences
    diffs = tf.abs(tf.expand_dims(X, 2) - tf.expand_dims(Y, 1))

    emd = tf.math.reduce_sum(factors * diffs, axis=[1, 2]) / tf.reduce_sum(factors)

    return tf.math.reduce_mean(emd)



def hellinger_distance(x, y):
    """
    Computes the Hellinger distance between two probability distributions https://en.wikipedia.org/wiki/Hellinger_distance.
    x, y - tensors of dimension (batch_size, n_classes), containing the logits for natural and adversarial samples, respectively.
    x, y are assumed to be already transformed into probability distributions
    """
    sqrt_2 = tf.constant(tf.math.sqrt(2.0))
    """tf.norm is very unstable for low values of argument - it would return d(norm(x))/dx, and for small values it gets 0/0 -> nan gradients"""
    # norms = tf.norm(tf.math.sqrt(x) - tf.math.sqrt(y), ord=2, axis=-1) / sqrt_2
    norms = tf.math.sqrt(tf.reduce_sum(tf.square(tf.math.sqrt(x) - tf.math.sqrt(y)), axis=-1) + tf.keras.backend.epsilon()) / sqrt_2

    return tf.math.reduce_mean(norms)


def bhatt_distance(x, y):
    """
    Computes the Bhattacharyya  distance between two probability distributions https://en.wikipedia.org/wiki/Bhattacharyya_distance.
    x, y - tensors of dimension (batch_size, n_classes), containing the logits for natural and adversarial samples, respectively.
    x, y are assumed to be already transformed into probability distributions
    """

    bhatt_coeff = tf.math.reduce_sum(tf.math.sqrt(x * y + tf.keras.backend.epsilon()), axis=-1)
    bhatt_dists = - tf.math.log(bhatt_coeff)

    return tf.math.reduce_mean(bhatt_dists)



def trades_loss(
        y_true, 
        logits_natural, 
        logits_adversarial,
        delta,
        dist="kl"
):
    """
    y_true: int, the correct class
    logits_natural/adversarial: vector of lentgh n_classes, the pre-activation output of the model
    """
    
    class_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    if dist == "kl":
        # Kullback Leibler
        prob_loss = tf.keras.losses.KLDivergence()
    elif dist == "jsd":
        # Jensen-Shannon
        prob_loss = lambda x, y: jensen_shannon(x, y)
    elif dist == "hell":
        # Hellinger
        prob_loss = lambda x, y: hellinger_distance(x, y)
    elif dist == "bhatt":
        # Bhattacharya
        prob_loss = lambda x, y: bhatt_distance(x, y)
    elif dist == "emd":
        # Earth Mover Distance
        prob_loss = lambda x, y: earth_mover_distance(x, y)
    else:
        raise ValueError(f"Distance {dist} unknown.")

    loss1 = class_loss(y_true, logits_natural)
    loss2 = prob_loss(
        tf.nn.softmax(logits_natural, axis=-1), 
        tf.nn.softmax(logits_adversarial, axis=-1)
    )

    return loss1, loss2


if __name__ == "__main__":

    ###
    dataset_type = 3
    batch_size = 512
    distance = "emd" # distance to measure the difference between probability distributions

    if dataset_type in [1, 2]:
        epsilon = 1.0
    elif dataset_type == 3:
        epsilon = 0.5
    elif dataset_type == 4:
        epsilon = 0.3
    elif dataset_type == 5:
        epsilon = 1.0
    ###

    dset_name = datasets[dataset_type]

    x_train, y_train, x_test, y_test, input_shape, n_classes = load_dataset(dataset_type)

    if dataset_type == 1:
        attack_steps = 50
    elif dataset_type == 2:
        attack_steps = 50
    elif dataset_type == 3:
        attack_steps = 30
    elif dataset_type == 4:
        attack_steps = 20
        batch_size = 128
    elif dataset_type == 5:
        attack_steps = 20
        batch_size = 128

    attacker = fb.attacks.L2ProjectedGradientDescentAttack(steps=attack_steps)

    for delta in [1e-2]:
        if distance != "kl":
            model_save_path = f"models_TRADES/{datasets[dataset_type]}/{distance}/delta={delta}_epsilon={epsilon}/"
        else:
            model_save_path = f"models_TRADES/{datasets[dataset_type]}/delta={delta}/epsilon={epsilon}/"
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
        elif dataset_type == 5:
            epochs = 100
            model = celeba_model(input_shape, n_classes)
        else:
            raise ValueError("Not implemented yet.")

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        for epoch in range(epochs):
            epoch_loss_clsf = 0
            epoch_loss_adv = 0

            "Clone model to be attacked"
            foolbox_model = fb.models.TensorFlowModel(model, bounds=(-1, 1))
            # Iterate over the batches of the dataset.
            if epoch != 0:
                _, clipped_adv, _ = make_adv_batch(foolbox_model, 
                                                lambda foolbox_model, x, y, epsilons: attacker(foolbox_model, x, y, epsilons=epsilons), 
                                                epsilon, x_train, y_train, batch_size=256)
            else:
                clipped_adv = x_train.copy()
            for i in tqdm(range(x_train.shape[0] // batch_size)):
                x_batch_train = x_train[i * batch_size: (i+1) * batch_size]
                y_batch_train = y_train[i * batch_size: (i+1) * batch_size]
                clipped_adv_i = clipped_adv[i * batch_size: (i+1) * batch_size]
                #  clipped_adv = x_batch_train
                with tf.GradientTape(persistent=True) as tape:
                    logits_adv = model(clipped_adv_i, training=False)
                    logits = model(x_batch_train, training=True)  # Logits for this minibatch
                    loss_cls, loss_kl = trades_loss(
                        y_batch_train, 
                        logits,
                        logits_adv,
                        delta=delta,
                        dist=distance
                    )
                    loss_value = loss_cls + delta * loss_kl
                # grads_kl = tape.gradient(loss_kl, model.trainable_weights)
                grads = tape.gradient(loss_value, model.trainable_weights)
                
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                epoch_loss_clsf += loss_cls
                epoch_loss_adv += loss_kl

            epoch_loss_clsf /= (x_train.shape[0] // batch_size)
            epoch_loss_adv /= (x_train.shape[0] // batch_size)

            y_test_pred = model(x_test)
            test_acc_now = np.mean(np.argmax(y_test_pred, axis=-1) == y_test)
            print(f"Epoch {epoch}/{epochs} train loss: classification={np.round(epoch_loss_clsf, 4)} adversarial={np.round(epoch_loss_adv, 4)}\n \
                    Test accuracy={test_acc_now}")

            model.save_weights(model_save_path + "model.hdf5")
