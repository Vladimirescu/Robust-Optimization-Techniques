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
from utils import Lion, load_dataset


datasets = {
    1: "MNIST",
    2: "FMNIST",
    3: "CIFAR-10",
    4: "eurosat",
    5: "CelebA"
}


def macer_loss(
        y_true, 
        logits, 
        beta,
        gamma,
        sigma
):
    """
    y_true: int, the correct class
    logits_natural/adversarial: vector of lentgh n_classes, the pre-activation output of the model
    """
    class_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    normal_distr = tf.compat.v1.distributions.Normal(loc=0.0, scale=1.0)

    # classification loss
    outputs_soft = tf.math.reduce_mean(tf.nn.softmax(logits, axis=2), axis=1)
    loss1 = class_loss(y_true, outputs_soft)
    
    # robust loss
    beta_outputs = logits * beta
    beta_outputs_soft = tf.math.reduce_mean(tf.nn.softmax(beta_outputs, axis=2), axis=1)

    top2 = tf.math.top_k(beta_outputs_soft, k=2)
    top2_score = top2.values
    top2_indices = top2.indices
    idx_correct = top2_indices[:, 0] == y_true

    out0 = top2_score[idx_correct][:, 0]
    out1 = top2_score[idx_correct][:, 1]
    rob_loss = normal_distr.quantile(out1) - normal_distr.quantile(out0)
    rob_loss = rob_loss[tf.math.abs(rob_loss) <= gamma] # only the parts that result in a positive hinge loss

    rob_loss = rob_loss + gamma

    loss2 = tf.reduce_sum(rob_loss) * sigma / 2.0

    return loss1, loss2


if __name__ == "__main__":

    ###
    dataset_type = 5
    batch_size = 256
  
    ###

    dset_name = datasets[dataset_type]

    x_train, y_train, x_test, y_test, input_shape, n_classes = load_dataset(dataset_type)

    if dataset_type == 1:
        k = 4
        gamma = 8 # hinge factor
        beta = 16 # inverse temperature
        sigmas = [0.05, 0.1, 0.2, 0.3, 0.4]
        lambds = [4, 5, 8, 10, 12]
    elif dataset_type == 2:
        k = 4
        gamma = 8 # hinge factor
        beta = 16 # inverse temperature
        # sigmas = [0.05, 0.1, 0.2, 0.3, 0.4]
        sigmas = [0.01]
        lambds = [1, 2, 3, 4]
    elif dataset_type == 3:
        k = 2
        gamma = 8 # hinge factor
        beta = 16 # inverse temperature
        sigmas = [0.3, 0.4]
        lambds = [0.01, 0.025, 0.05, 0.075, 0.1]
    elif dataset_type == 4:
        k = 2
        gamma = 8 # hinge factor
        beta = 16 # inverse temperature
        sigmas = [0.1]
        lambds = [0.1]
        epochs = 200
        batch_size = 128
    elif dataset_type == 5:
        k = 2
        gamma = 8 # hinge factor
        beta = 16 # inverse temperature
        sigmas = [0.1]
        lambds = [0.1]
        epochs = 200
        batch_size = 128

    for sigma in sigmas:
        for lambd in lambds:

            model_save_path = f"models_MACER/{datasets[dataset_type]}/sigma={sigma}/lambd={lambd}/"
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            if dataset_type in [1, 2]:
                epochs = 20
                model = model_AT_mnist(input_shape=input_shape, n_classes=n_classes, batch_norm=False)
            elif dataset_type == 3:
                epochs = 100
                model = model_AT_cifar(input_shape=input_shape, n_classes=n_classes, batch_norm=False)
            elif dataset_type == 4:
                epochs = 200
                model = res_net9(input_shape=input_shape, n_classes=n_classes, deel=True)
            elif dataset_type == 5:
                epochs = 200
                model = celeba_model(input_shape, n_classes)
            else:
                raise ValueError("Not implemented yet.")
    
            # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
            optimizer = Lion(learning_rate=1e-4)
            if dataset_type in [4, 5]:
                optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
            test_acc = -np.inf
            for epoch in range(epochs):
                epoch_loss_clsf = 0
                epoch_loss_rob = 0

                for i in tqdm(range(x_train.shape[0] // batch_size)):
                    x_batch_train = x_train[i * batch_size: (i+1) * batch_size]
                    y_batch_train = y_train[i * batch_size: (i+1) * batch_size]

                    # Repeat batch k times to create k noisy versions
                    x_batch_train_k = np.repeat(x_batch_train, k, axis=0)
                    x_batch_train_k_noisy = x_batch_train_k + np.random.normal(0, sigma, x_batch_train_k.shape)

                    with tf.GradientTape(persistent=True) as tape:
                        logits = model(x_batch_train_k_noisy, training=True)  # Logits for this minibatch
                        logits = tf.reshape(logits, (batch_size, k, -1))
                        loss_cls, loss_rob = macer_loss(
                            y_batch_train, logits, beta, gamma, sigma
                        )

                        loss_value = loss_cls + lambd * loss_rob
                    grads = tape.gradient(loss_value, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
                    epoch_loss_clsf += loss_cls
                    epoch_loss_rob += loss_rob

                    grads_rob = tape.gradient(loss_rob, model.trainable_weights)

                epoch_loss_clsf /= (x_train.shape[0] // batch_size)
                epoch_loss_rob /= (x_train.shape[0] // batch_size)

                y_test_pred = model(x_test)
                test_acc_now = np.mean(np.argmax(y_test_pred, axis=-1) == y_test)
                print(f"Epoch {epoch}/{epochs} train loss: classification={np.round(epoch_loss_clsf, 4)} robustness={np.round(epoch_loss_rob, 4)}\n \
                        Test accuracy={test_acc_now}")
                if test_acc_now > test_acc:
                    test_acc = test_acc_now
                    model.save_weights(model_save_path + "model.hdf5")
