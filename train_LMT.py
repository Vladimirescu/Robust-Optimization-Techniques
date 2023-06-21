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
from utils import load_dataset, LMT, SingularValueClip


datasets = {
    1: "MNIST",
    2: "FMNIST",
    3: "CIFAR-10",
    4: "eurosat",
    5: "CelebA",
    6: "BelgiumTS",
}


if __name__ == "__main__":

    ###
    dataset_type = 6
    batch_size = 512

    dset_name = datasets[dataset_type]

    x_train, y_train, x_test, y_test, input_shape, n_classes = load_dataset(dataset_type)

    for c in [500]:
        model_save_path = f"models_LMT/{datasets[dataset_type]}/c={c}/"
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if dataset_type in [1, 2]:
            epochs = 20
            model = model_AT_mnist(input_shape=input_shape, n_classes=n_classes, batch_norm=False)
        elif dataset_type == 3:
            epochs = 50
            model = model_AT_cifar(input_shape=input_shape, n_classes=n_classes, batch_norm=False)
        elif dataset_type == 4:
            epochs = 100
            model = res_net9(input_shape=input_shape, n_classes=n_classes, deel=True)
        elif dataset_type == 5:
            epochs = 100
            model = celeba_model(input_shape, n_classes)
        elif dataset_type == 6:
            epochs = 100
            model = res_net9(input_shape, n_classes, filters_start=8, deel=True)
            model.load_weights("models_LMT/BelgiumTS/c=500/model.hdf5")
        else:
            raise ValueError("Not implemented yet.")

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        if dataset_type in [4, 5, 6]:
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=0.5)

        """Initialize callback to compute sub-Lip and constrain last dense"""
        svd = SingularValueClip(lip=0.4)
        svd.set_model(model)
        svd.on_train_begin()

        lmt = LMT(lip_const_last=False)
        lmt.set_model(model)
        lmt.on_train_begin()

        for epoch in range(epochs):
            epoch_loss = 0

            for i in tqdm(range(x_train.shape[0] // batch_size)):
                x_batch_train = x_train[i * batch_size: (i+1) * batch_size]
                y_batch_train = y_train[i * batch_size: (i+1) * batch_size]

                svd.on_batch_end(0)
                lip_const = lmt.get_lip()
                # print(lip_const)

                with tf.GradientTape(persistent=True) as tape:
                    logits = model(x_batch_train, training=True)  # Logits for this minibatch

                    mask_keep = tf.one_hot(y_batch_train, depth=n_classes)
                    mask_add = 1 - mask_keep

                    logits_lmt = mask_keep * logits + mask_add * (logits + tf.math.sqrt(2.0) * c * lip_const)

                    loss_value = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_batch_train, logits_lmt)

                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                epoch_loss += loss_value

            epoch_loss /= (x_train.shape[0] // batch_size)
            y_test_pred = model(x_test)
            test_acc_now = np.mean(np.argmax(y_test_pred, axis=-1) == y_test)
            print(f"Epoch {epoch}/{epochs} Train loss: {epoch_loss} Test accuracy={test_acc_now}")

            model.save_weights(model_save_path + "model.hdf5")
