import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from models import model_AT_mnist, model_AT_cifar, res_net9, celeba_model
from AT_utils import make_adv_batch
import foolbox as fb
import numpy as np
from tqdm import tqdm
from utils import Lion, load_dataset


datasets = {
    1: "MNIST",
    2: "FMNIST",
    3: "CIFAR-10",
    4: "eurosat",
    5: "CelebA"
}


def manual_pgd_attack(x_batch, y_batch, model, n_steps, step_size):
    x_batch_adv = tf.cast(x_batch, float)
    loss_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    for _ in range(n_steps):
        with tf.GradientTape() as tape_x:
            tape_x.watch(x_batch_adv)
            logits = model(x_batch_adv)
            loss = loss_cls(y_batch, logits)

        gradient = tape_x.gradient(loss, x_batch_adv)
        sign_batch = tf.sign(gradient)

        x_batch_adv = tf.clip_by_value(x_batch_adv + step_size * sign_batch, -1 , 1)
    
    return x_batch_adv


def stable_norm2(x, axis):
    return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(x), axis=axis))


def deep_defense_reg(
        y_true,
        logits,
        x_batch,
        x_batch_adv,
        c,
        d
):
    x_batch = tf.cast(x_batch, float)
    
    delta_perturb = stable_norm2(
        tf.reshape(x_batch, (x_batch.shape[0], -1)) - tf.reshape(x_batch_adv, (x_batch_adv.shape[0], -1)), 
        axis=-1)
    delta_x = stable_norm2(
        tf.reshape(x_batch, (x_batch.shape[0], -1)), 
        axis=-1)

    # correct_cls_mask = tf.argmax(logits, axis=-1) == y_true
    # sincorrect_cls_mask = tf.math.logical_not(correct_cls_mask)

    # correct_terms = delta_perturb[correct_cls_mask] / (delta_x[correct_cls_mask] + tf.keras.backend.epsilon())
    # incorrect_terms = delta_perturb[incorrect_cls_mask] / (delta_x[incorrect_cls_mask] + tf.keras.backend.epsilon())
    all_terms = delta_perturb / (delta_x + tf.keras.backend.epsilon())

    # reg = tf.math.reduce_sum(tf.math.exp(-c * correct_terms)) + tf.math.reduce_sum(tf.math.exp(d * incorrect_terms))
    # reg = tf.math.reduce_mean(-c * correct_terms) + tf.math.reduce_mean(d * incorrect_terms)
    # reg = tf.math.reduce_sum(tf.math.exp(-c * correct_terms))
    # reg = tf.math.reduce_sum(- delta_perturb / delta_x)
    reg = tf.reduce_mean(tf.math.exp(-c * all_terms))
    # reg = tf.math.reduce_mean(1 / (c * correct_terms + tf.keras.backend.epsilon()))
    # reg = tf.math.reduce_mean(1 / (c * correct_terms + tf.keras.backend.epsilon())) 
    # reg = tf.math.reduce_mean(c * correct_terms)

    return reg


if __name__ == "__main__":

    ###
    dataset_type = 5
    batch_size = 512
    d = 1

    if dataset_type == 1:
        attack_steps = 1
        attack_step_size = 0.5
        epochs = 30
        batch_size = 1024
    elif dataset_type == 2:
        attack_steps = 1
        attack_step_size = 1
        epochs = 30
    elif dataset_type == 3:
        epochs = 50
        attack_steps = 1
        attack_step_size = 0.25
    elif dataset_type == 4:
        epochs = 50
        batch_size = 128
        attack_steps = 1
        attack_step_size = 0.3
    elif dataset_type == 4:
        epochs = 50
        batch_size = 128
        attack_steps = 1
        attack_step_size = 0.3
    elif dataset_type == 5:
        epochs = 50
        batch_size = 128
        attack_steps = 1
        attack_step_size = 0.1
    

    for factor in [0.1]:
        for c in [0.01]:
            # model_save_path = f"models_DeepDefense/{datasets[dataset_type]}/at_c={c}_lambd={lambd}_1step_grad/"
            model_save_path = f"models_DeepDefense/{datasets[dataset_type]}/lambda/factor={factor}_c={c}_exp/"
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            dset_name = datasets[dataset_type]
            x_train, y_train, x_test, y_test, input_shape, n_classes = load_dataset(dataset_type)

            if dataset_type in [1, 2]:
                model = model_AT_mnist(input_shape=input_shape, n_classes=n_classes, batch_norm=False)
            elif dataset_type == 3:
                model = model_AT_cifar(input_shape=input_shape, n_classes=n_classes, batch_norm=False)
            elif dataset_type == 4:
                model = res_net9(input_shape=input_shape, n_classes=n_classes, deel=True)
            elif dataset_type == 5:
                model = celeba_model(input_shape, n_classes)
            else:
                raise ValueError("Not implemented yet.")

            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
            if dataset_type in [4, 5]:
                optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
            best_loss = np.inf

            lambd = 1
            for epoch in range(epochs):
                epoch_loss_clsf = 0
                epoch_reg_term = 0

                for i in tqdm(range(x_train.shape[0] // batch_size)):
                    x_batch_train = x_train[i * batch_size: (i+1) * batch_size] 
                    y_batch_train = y_train[i * batch_size: (i+1) * batch_size]

                    x_batch_train = tf.cast(x_batch_train, float)
                    y_batch_train = tf.cast(y_batch_train, tf.int64)
                    class_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                    with tf.GradientTape(persistent=True) as tape:
                        logits = model(x_batch_train, training=True)  # Logits and embeddings for this positives

                        "Compute adversarial examples using PGD"
                        x_batch_adv = tf.identity(x_batch_train)
                        tape.watch(x_batch_adv)
                        for _ in range(attack_steps):
                            logits_adv = model(x_batch_adv)
                            loss_adv = class_loss(y_batch_train, logits_adv)
                            grad_x = tape.gradient(loss_adv, x_batch_adv)
                            grad_x_normed = 2 * (grad_x - tf.math.reduce_min(grad_x)) / (tf.math.reduce_max(grad_x) - tf.math.reduce_min(grad_x) + tf.keras.backend.epsilon()) - 1
                            x_batch_adv = tf.clip_by_value(x_batch_adv + attack_step_size * grad_x_normed, -1, 1)

                        # x_batch_adv_reg = tf.identity(x_batch_adv)
                        # """Compute regularization term"""
                        reg = deep_defense_reg(y_batch_train, 
                                                            logits, 
                                                            x_batch_train,
                                                            x_batch_adv,
                                                            c, d)

                        # loss_cls = class_loss(y_batch_train, logits)
                        loss_cls = class_loss(y_batch_train, model(x_batch_adv))
                        loss_total = loss_cls + factor * reg

                    # print(loss_cls, lambd * reg)
                    grads = tape.gradient(loss_total, model.trainable_variables)

                    # grads_c = tape.gradient(loss_cls, model.trainable_variables)
                    # grads_r = tape.gradient(reg, model.trainable_variables)

                    # norm_out_c = tf.linalg.norm(grads_c[0])
                    # norm_out_r = tf.linalg.norm(grads_r[0])
                    # epoch_grad_frac += norm_out_c / norm_out_r
                    
                    """Manually compute overall gradients in order to account for smaller magnitudes of grads_r"""
                    # grads = []
                    # for i in range(len(grads_c)):
                    #     grads_r[i] = grads_r[i] / tf.linalg.norm(grads_r[i])
                    #     factor = lambd * tf.linalg.norm(grads_c[i])

                    #     grads_total = grads_c[i] + grads_r[i] * factor
                    #     grads.append(grads_total)

                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
                    
                    epoch_loss_clsf += loss_cls
                    epoch_reg_term += reg

                epoch_loss_clsf /= (x_train.shape[0] // batch_size)
                epoch_reg_term /= (x_train.shape[0] // batch_size)

                """Adapt lambda such that the factor between losses is the desired one"""
                lambd = epoch_loss_clsf / (epoch_reg_term * factor + tf.keras.backend.epsilon())

                print(f"Changing lambda to {lambd}")

                y_train_pred = model.predict(x_train, batch_size=256)
                y_test_pred = model.predict(x_test, batch_size=256)

                train_acc_now = np.mean(np.argmax(y_train_pred, axis=-1) == y_train)
                test_acc_now = np.mean(np.argmax(y_test_pred, axis=-1) == y_test)

                test_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_test, y_test_pred)

                print(f"Epoch {epoch + 1}/{epochs} train loss: classification={np.round(epoch_loss_clsf, 4)} reg={np.round(epoch_reg_term, 4)}\
                     \n Train accuracy={train_acc_now} Test accuracy={test_acc_now}")
                if test_loss < best_loss:
                    best_loss = test_loss
                    model.save_weights(model_save_path + "model.hdf5")