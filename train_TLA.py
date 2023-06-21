import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
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


def make_negatives_positives(x, y, typ="neg"):
    """
    Either randomly sample positives or negatives samples for each x_i.

    x: array of shape (n_samples, ....) - dtype = float
    y: array of shape (n_samples,) - dtype = int

    Generates a versions of x (same shape, same data): x_neg, which represent a different shuffle of x, s.t.:
    - for each i = 1,...,n_samples: label(x[i]) != label(x_neg[i]) (if typ = neg)
    - for each i = 1,...,n_samples: label(x[i]) == label(x_neg[i]) (if typ = pos)

    i.e. shuffle y s.t. at each position the value changes, and record the indexes.

    Note: the current method may yield the same negative/positive example multiple times, and may exclude some of the training data
    """

    indexes = np.arange(y.shape[0])
    indexes_neg = np.empty_like(indexes)

    for i in range(indexes_neg.shape[0]):
        if typ == "neg":
            indexes_neg[i] = np.random.choice(indexes[y != y[i]]) # randomly choose negative
        elif typ == "pos":
            indexes_neg[i] = np.random.choice(indexes[y == y[i]]) # randomly choose positive
        else:
            raise ValueError("Unknown value for parameter typ.")

    return x[indexes_neg]


def triplet_loss(pos, anch, neg, margin=1.0):
    """
    pos, anch, neg: arrays of shape (batch_size, n_embedding), representing the positive, anchor and negative samples    
    margin: triplet margin for distance
    """
    pos = tf.cast(pos, float)
    anch = tf.cast(anch, float)
    neg = tf.cast(neg, float)

    pos = pos / tf.norm(pos, axis=-1, keepdims=True)
    anch = anch / tf.norm(anch, axis=-1, keepdims=True)
    neg = neg / tf.norm(neg, axis=-1, keepdims=True)

    """
    tf.keras.losses.cosine_similarity(x, y) = 1 when x and y are the most DISSIMILAR
                                            = -1 when x and y are the most SIMILAR
    This way can be used as a loss, but we need it as a distance
    """

    # this way, the distances are restricted in the range [0, 1]
    d1 = (1 + tf.keras.losses.cosine_similarity(pos, anch, axis=-1)) / 2
    d2 = (1 + tf.keras.losses.cosine_similarity(anch, neg, axis=-1)) / 2

    loss = d1 - d2 + margin
    loss = tf.reduce_sum(tf.maximum(loss, 0))

    avg_distance = tf.math.reduce_mean(d2 - d1)

    return loss, avg_distance


def tla_loss(
        y_true, 
        logits_natural, 
        embeddings_natural,
        embeddings_adversarial,
        embeddings_negative,
        margin
):
    """
    y_true: int, correct classes
    logits_natural/adversarial: logits of natural/adversarial examples
    embeddings_natural/adversarial/negative: embedding vectors for natural examples, and corresponding generated adversarial exampels; negative refers
    to the embedding of the selected example for each natural example
    """
    class_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    loss1 = class_loss(y_true, logits_natural)
    loss2, avg_dist = triplet_loss(embeddings_natural, embeddings_adversarial, embeddings_negative, margin=margin)
    loss3 = tf.norm(embeddings_natural, ord=2, axis=-1) + \
            tf.norm(embeddings_adversarial, ord=2, axis=-1) + \
            tf.norm(embeddings_negative, ord=2, axis=-1)
    loss3 = tf.reduce_sum(loss3)

    return loss1, loss2, loss3, avg_dist


if __name__ == "__main__":

    ###
    dataset_type = 3
    batch_size = 64
    attack_steps = 20
    attacker = fb.attacks.L2ProjectedGradientDescentAttack(steps=attack_steps)
    lambd_2 = 1e-4
    ###

    dset_name = datasets[dataset_type]

    x_train, y_train, x_test, y_test, input_shape, n_classes = load_dataset(dataset_type)

    if dataset_type == 1:
        epsilon = 2.0
    elif dataset_type == 2:
        epsilon = 2.0
    elif dataset_type == 3:
        epsilon = 1.0
    elif dataset_type == 4:
        epsilon = 0.5
        batch_size = 32
    elif dataset_type == 5:
        epsilon = 0.5
        batch_size = 32

    lambd_1_s = [0.5, 1, 2]
    margins = [0.1]
    for lambd_1 in lambd_1_s:
        for margin in margins:
            model_save_path = f"models_TLA/{datasets[dataset_type]}/lambd={lambd_1}/margin={margin}/"
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            if dataset_type in [1, 2]:
                epochs = 20
                model = model_AT_mnist(input_shape=input_shape, n_classes=n_classes, return_embeddings=True, batch_norm=False)
                model_no_embedd = model_AT_mnist(input_shape=input_shape, n_classes=n_classes, batch_norm=False)
            elif dataset_type == 3:
                epochs = 30
                model = model_AT_cifar(input_shape=input_shape, n_classes=n_classes, return_embeddings=True, batch_norm=False)
                model_no_embedd = model_AT_cifar(input_shape=input_shape, n_classes=n_classes, batch_norm=False)
            elif dataset_type == 4:
                epochs = 100
                model = res_net9(input_shape=input_shape, n_classes=n_classes, return_embeddings=True, deel=True)
                model_no_embedd = res_net9(input_shape=input_shape, n_classes=n_classes, deel=True)
            elif dataset_type == 5:
                epochs = 100
                model = celeba_model(input_shape, n_classes, return_embeddings=True)
                model_no_embedd = celeba_model(input_shape, n_classes)
            else:
                raise ValueError("Not implemented yet.")
            model.summary()

            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
            test_acc = -np.inf
            exp_margin_list = []

            for epoch in range(epochs):
                epoch_loss_clsf = 0
                epoch_loss_triplet = 0
                epoch_loss_norm = 0
                epoch_avg_margin = 0

                "Clone model to be attacked - need to use a model that does not also return the embeddings -- FoolBox only accept one output"
                model_no_embedd.set_weights(model.get_weights())
                foolbox_model = fb.models.TensorFlowModel(model_no_embedd, bounds=(-1, 1))
                # Iterate over the batches of the dataset and compute adv examples for the current epoch
                _, clipped_adv, _ = make_adv_batch(foolbox_model, 
                                                lambda foolbox_model, x, y, epsilons: attacker(foolbox_model, x, y, epsilons=epsilons), 
                                                epsilon, x_train, y_train, batch_size=256)
                
                """Generate (positive, anchor, negative) pairs - random selection for negative samples. Anchor = clipped_adv
                Negatives are randomly selected as examples from different class than positives, at the beginning of each epoch.

                Also, select positive samples as randomly sampled images with the same class for each index.
                """
                negatives = make_negatives_positives(x_train, y_train, typ="neg")
                positives = make_negatives_positives(x_train, y_train, typ="pos")

                for i in tqdm(range(x_train.shape[0] // batch_size)):
                    x_batch_train_pos = positives[i * batch_size: (i+1) * batch_size] # batch pos
                    x_batch_train_neg = negatives[i * batch_size: (i+1) * batch_size] # batch neg
                    y_batch_train = y_train[i * batch_size: (i+1) * batch_size]
                    clipped_adv_i = clipped_adv[i * batch_size: (i+1) * batch_size] # batch anchor

                    with tf.GradientTape() as tape:
                        logits_adv, embeddings_adv = model(clipped_adv_i, training=True) # embeddings for adversarials
                        _, embeddings_pos = model(x_batch_train_pos, training=True)  # Logits and embeddings for this positives
                        _, embeddings_neg = model(x_batch_train_neg, training=True) # Embeddings for negative samples
                        loss_cls, loss_triplet, loss_norm, avg_margin = tla_loss(y_batch_train, 
                                                                    logits_adv, 
                                                                    embeddings_pos,
                                                                    embeddings_adv,
                                                                    embeddings_neg,
                                                                    margin)
                        loss_value = loss_cls + lambd_1 * loss_triplet + lambd_2 * loss_norm

                    grads = tape.gradient(loss_value, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
                    epoch_loss_clsf += loss_cls
                    epoch_loss_triplet += loss_triplet
                    epoch_loss_norm += loss_norm
                    epoch_avg_margin += avg_margin

                epoch_loss_clsf /= (x_train.shape[0] // batch_size)
                epoch_loss_triplet /= (x_train.shape[0] // batch_size)
                epoch_loss_norm /= (x_train.shape[0] // batch_size)
                epoch_avg_margin /= (x_train.shape[0] // batch_size)

                "Save the current list of average margin"
                exp_margin_list.append(epoch_avg_margin)
                np.save(model_save_path + "avg_margin_epoch.npy", exp_margin_list)

                y_train_pred = model.predict(x_train, batch_size=256)[0]
                y_test_pred = model.predict(x_test, batch_size=256)[0]

                train_acc_now = np.mean(np.argmax(y_train_pred, axis=-1) == y_train)
                test_acc_now = np.mean(np.argmax(y_test_pred, axis=-1) == y_test)
                
                print(f"Epoch {epoch}/{epochs} train loss: classification={np.round(epoch_loss_clsf, 4)} triplet={np.round(epoch_loss_triplet, 4)} \
                    norm={np.round(epoch_loss_norm, 4)}  avg_margin={np.round(avg_margin, 4)} \n Train accuracy={train_acc_now} Test accuracy={test_acc_now}")
                
                # if test_acc_now > test_acc:
                #     test_acc = test_acc_now
                """Save the last model -- this makes sure that both loss objectives were optimized"""
                model_no_embedd.set_weights(model.get_weights())
                model_no_embedd.save_weights(model_save_path + "model.hdf5")