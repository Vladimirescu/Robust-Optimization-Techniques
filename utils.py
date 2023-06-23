import tensorflow as tf
import tensorflow.compat.v2 as tf
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds


def load_dataset(dataset_type):
    if dataset_type == 1:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = (x_train - 127.5) / 127.5
        x_test = (x_test - 127.5) / 127.5
        input_shape = (28, 28, 1)
        n_classes = 10
    elif dataset_type == 2:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = (x_train - 127.5) / 127.5
        x_test = (x_test - 127.5) / 127.5
        input_shape = (28, 28, 1)
        n_classes = 10
    elif dataset_type == 3:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        x_train = (x_train - 127.5) / 127.5
        x_test = (x_test - 127.5) / 127.5
        input_shape = (32, 32, 3)
        n_classes = 10
    elif dataset_type == 4:
        ds, ds_info = tfds.load('eurosat/rgb',
                            with_info=True,
                            split='train',
                            data_dir="/scratch/")
        
        X = []
        y = []
        for i, x in enumerate(ds):
            X.append(x["image"].numpy().astype(float))
            y.append(x["label"].numpy().astype(int))

        X = (np.array(X) - 127.5) / 127.5
        y = np.array(y)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

        input_shape = (64, 64, 3)
        n_classes = 10
    elif dataset_type == 5:
        input_shape = (128, 128, 3)
        n_classes = 2

        train_ds = tf.keras.utils.image_dataset_from_directory(
            "../CelebA_smaller/train/", 
            label_mode="int", 
            color_mode="rgb",
            image_size=(input_shape[0], input_shape[1]),
            batch_size=128)
        test_ds = tf.keras.utils.image_dataset_from_directory(
            "../CelebA_smaller/test/", 
            label_mode="int", 
            color_mode="rgb",
            image_size=(input_shape[0], input_shape[1]),
            batch_size=128)
        
        rescaling = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)
        train_ds = train_ds.map(lambda x,y : (rescaling(x),y))
        test_ds = test_ds.map(lambda x,y : (rescaling(x),y))

        x_train = None
        y_train = None
        for i, (x, y) in enumerate(train_ds):
            if x_train is None:
                x_train = x
                y_train = y
            else:
                x_train = np.concatenate((x_train, x), axis=0)
                y_train = np.concatenate((y_train, y), axis=0)

        x_test = None
        y_test = None
        for i, (x, y) in enumerate(test_ds):
            if x_test is None:
                x_test = x
                y_test = y
            else:
                x_test = np.concatenate((x_test, x), axis=0)
                y_test = np.concatenate((y_test, y), axis=0)
    else:
       raise ValueError(f"Dataset type {dataset_type} not implemented.")
    
    return x_train, y_train, x_test, y_test, input_shape, n_classes
  

class SingularValueClip(tf.keras.callbacks.Callback):
    def __init__(self, lip, free_lips=False):
        """
        lip: depending on free_lip, it has two meanings:
            free_lip = False -- lip represents the per-layer lip to be imposed
            free_lip = True -- lip^{n_layers} represents the total lip to be imposed to the network; this will
            be done by splitting the value of lip into lips for each layer, according to their current influence
        """

        self.lip = lip
        self.free_lips = free_lips

        self.dense_layers = []
        self.conv_layers = []

    def get_dense_conv_lips(self):
        """
        Returns two lists of Lipschitz values: 1 list for dense layers, 1 list for conv layers
        """
        dense_lips = []
        conv_lips = []
        for x in self.dense_layers:
          w, b = x.get_weights()
          s = tf.linalg.svd(w, compute_uv=False)
          dense_lips.append(s[0])
        for i, x in enumerate(self.conv_layers):
          w, b = x.get_weights()
          w = tf.reshape(w, [-1, w.shape[-1]])
          s =tf.linalg.svd(w, compute_uv=False)
          conv_lips.append(s[0])

        return np.array(dense_lips), np.array(conv_lips)

    def on_batch_end(self, batch, logs=None):
        
        if self.free_lips:
           """Compute, for each layer, the Lipschitz constant to constrain it to, such that:
           prod Lipschitz constants = self.lip ^ n_layers
           """
           dense_lips, conv_lips = self.get_dense_conv_lips()
           lip_max = self.lip**self.n_layers
           lip_now = np.prod(dense_lips) * np.prod(conv_lips)
           
           """compute scaling factor to impose product = lip_max"""
           scale = (lip_max / lip_now)**(1 / self.n_layers)

           dense_lips = dense_lips * scale
           conv_lips = conv_lips * scale
        else:
           dense_lips = [self.lip] * len(self.dense_layers)
           conv_lips = [self.lip] * len(self.conv_layers)

        """Constrain dense layers"""
        for i, x in enumerate(self.dense_layers):
            w, b = x.get_weights()

            s, u, v = tf.linalg.svd(w)
            s_clipped = tf.clip_by_value(s, 0, dense_lips[i])
            w_clipped = tf.matmul(u, tf.matmul(tf.linalg.diag(s_clipped), v, adjoint_b=True))

            x.set_weights([w_clipped, b])

        """Constrain conv layers"""
        for i, x in enumerate(self.conv_layers):
            w, b = x.get_weights()

            w_clipped = tf.reshape(w, [-1, w.shape[-1]])

            s, u, v = tf.linalg.svd(w_clipped)
            s_clipped = tf.clip_by_value(s, 0, conv_lips[i])
            w_clipped = tf.matmul(u, tf.matmul(tf.linalg.diag(s_clipped), v, adjoint_b=True))

            w_clipped = tf.reshape(w_clipped, w.shape)

            x.set_weights([w_clipped, b])


    def on_train_begin(self, logs=None):

        for x in self.model.layers:
            if isinstance(x, tf.keras.layers.Dense):
                self.dense_layers.append(x)
            if isinstance(x, tf.keras.layers.Conv2D):
                self.conv_layers.append(x)

        self.n_layers = len(self.dense_layers) + len(self.conv_layers)

