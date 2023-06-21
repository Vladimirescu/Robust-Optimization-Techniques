import tensorflow as tf
from deel.lip.layers import SpectralDense, SpectralConv2D


@tf.custom_gradient
def custom_round(x):
    r = tf.round(x)
    def grad(upstream):
        return 1
    return r, grad


class QuantizeWeights(tf.keras.constraints.Constraint):
    
     def __init__(self, n_bits=32):
        """Number of bits to quantize the weight/bias tensor."""
        super().__init__()
        self.n_bits = n_bits

     def __call__(self, w):
        """Different quantization scheme from EMPIR"""
        n = float(2 ** self.n_bits - 1)

        max_init = tf.reduce_max(w)
        min_init = tf.reduce_min(w)

        w_quantized = (w - min_init) / (max_init - min_init + tf.keras.backend.epsilon())
        w_quantized = custom_round(w_quantized * n) / n

        return (max_init - min_init) * w_quantized + min_init

     def get_config(self):
         return {"n_bits": self.n_bits}
          

def quantizedReLU(x, n_bits):
    """Different quantization scheme from EMPIR"""
    n = float(2 ** n_bits - 1)
    relu = tf.nn.relu(x) * n
    # return custom_round(relu) / n
    return tf.nn.relu(x)


def model_AT_mnist(input_shape, n_classes=10, return_embeddings=False, batch_norm=True):
    """
    :param return_embeddings: whether to return intermediat features -- for TLA
    :param batch_norm: whether utilize BatchNormalization, or Dropout -- for GloroNet use Dropout, 
    since BatchNorm modifies the Lipschitz constant
    """

    inpt = tf.keras.layers.Input(shape=input_shape)

    hdn = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')(inpt)
    hdn = tf.keras.activations.relu(hdn)
    if batch_norm:
        hdn = tf.keras.layers.BatchNormalization()(hdn)
    else:
        hdn = tf.keras.layers.Dropout(rate=0.1)(hdn)
    hdn = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(hdn)
    hdn = tf.keras.activations.relu(hdn)
    if batch_norm:
        hdn = tf.keras.layers.BatchNormalization()(hdn)
    else:
        hdn = tf.keras.layers.Dropout(rate=0.1)(hdn)
    hdn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(hdn)

    hdn = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(hdn)
    hdn = tf.keras.activations.relu(hdn)
    if batch_norm:
        hdn = tf.keras.layers.BatchNormalization()(hdn)
    else:
        hdn = tf.keras.layers.Dropout(rate=0.1)(hdn)
    hdn = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(hdn)
    hdn = tf.keras.activations.relu(hdn)
    if batch_norm:
        hdn = tf.keras.layers.BatchNormalization()(hdn)
    else:
        hdn = tf.keras.layers.Dropout(rate=0.1)(hdn)
    hdn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(hdn)

    hdn = tf.keras.layers.Flatten()(hdn)

    hdn = tf.keras.layers.Dense(256)(hdn)
    hdn = tf.keras.activations.relu(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.3)(hdn)

    embeddings = tf.keras.layers.Dense(128)(hdn)
    hdn = tf.keras.activations.relu(embeddings)
    hdn = tf.keras.layers.Dropout(rate=0.3)(hdn)

    out = tf.keras.layers.Dense(n_classes)(hdn)

    if return_embeddings:
        model = tf.keras.models.Model(inputs=inpt, outputs=[out, embeddings])
    else:
        model = tf.keras.models.Model(inputs=inpt, outputs=out)
    return model


def wrn_residualBlock(inputs, in_channels, out_channels, stride):
    if stride != 1 or in_channels != out_channels:
        skip_c = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1, strides=stride, padding='same')(inputs)
    else:
        skip_c = inputs

    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.Dropout(rate=0.1)(x)
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='same')(x)

    x = tf.add(skip_c, x)
        
    return x


def wide_resnet(input_shape=(32, 32, 3), num_classes=10, depth=28, widen_factor=10):
    """Inspired from https://github.com/akshaymehra24/WideResnet/blob/master/WRN_converted_to_tf.py"""
    n = int((depth-4)/6)
    k = widen_factor
    nStages = [16, 16*k, 32*k, 64*k]
    strides = [1, 1, 2, 2]

    inpt = tf.keras.layers.Input(shape=input_shape)
    hdn = inpt
    for i, stage in enumerate(nStages):
        if i == 0:
            hdn = tf.keras.layers.Conv2D(filters=nStages[0], kernel_size=3, strides=strides[0], padding='same')(hdn)
        else:
            """Create n residual blocks for each stage"""
            block_strides = [strides[i]] + [1] * (n-1)

            for j in range(n):
                hdn = wrn_residualBlock(hdn, nStages[i-1], stage, stride=block_strides[j])

    model = tf.keras.models.Model(inputs=inpt, outputs=hdn)
    return model



def model_AT_cifar(input_shape, n_classes=10, return_embeddings=False, batch_norm=True):
    """
    :param return_embeddings: whether to return intermediat features -- for TLA
    :param batch_norm: whether utilize BatchNormalization, or Dropout -- for GloroNet use Dropout, 
    since BatchNorm modifies the Lipschitz constant
    """

    inpt = tf.keras.layers.Input(shape=input_shape)

    ###
    hdn = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(inpt)
    hdn = tf.keras.activations.relu(hdn)
    if batch_norm:
        hdn = tf.keras.layers.BatchNormalization()(hdn)
    else:
        hdn = tf.keras.layers.Dropout(rate=0.1)(hdn)
    hdn = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(hdn)
    hdn = tf.keras.activations.relu(hdn)
    if batch_norm:
        hdn = tf.keras.layers.BatchNormalization()(hdn)
    else:
        hdn = tf.keras.layers.Dropout(rate=0.1)(hdn)
    hdn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(hdn)
    ###

    ###
    hdn = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same')(hdn)
    hdn = tf.keras.activations.relu(hdn)
    if batch_norm:
        hdn = tf.keras.layers.BatchNormalization()(hdn)
    else:
        hdn = tf.keras.layers.Dropout(rate=0.2)(hdn)
    hdn = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same')(hdn)
    hdn = tf.keras.activations.relu(hdn)
    if batch_norm:
        hdn = tf.keras.layers.BatchNormalization()(hdn)
    else:
        hdn = tf.keras.layers.Dropout(rate=0.2)(hdn)
    hdn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(hdn)
    ###

    ###
    hdn = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same')(hdn)
    hdn = tf.keras.activations.relu(hdn)
    if batch_norm:
        hdn = tf.keras.layers.BatchNormalization()(hdn)
    else:
        hdn = tf.keras.layers.Dropout(rate=0.3)(hdn)
    hdn = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same')(hdn)
    hdn = tf.keras.activations.relu(hdn)
    if batch_norm:
        hdn = tf.keras.layers.BatchNormalization()(hdn)
    else:
        hdn = tf.keras.layers.Dropout(rate=0.3)(hdn)
    hdn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(hdn)
    ###

    hdn = tf.keras.layers.Flatten()(hdn)

    hdn = tf.keras.layers.Dense(256)(hdn)
    hdn = tf.keras.activations.relu(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.4)(hdn)

    embeddings = tf.keras.layers.Dense(256)(hdn)
    hdn = tf.keras.activations.relu(embeddings)
    hdn = tf.keras.layers.Dropout(rate=0.3)(hdn)

    out = tf.keras.layers.Dense(n_classes)(hdn)

    if return_embeddings:
        model = tf.keras.models.Model(inputs=inpt, outputs=[out, embeddings])
    else:
        model = tf.keras.models.Model(inputs=inpt, outputs=out)
    return model


def model_AT_cifar_deelLip(input_shape, n_classes=10, return_embeddings=False, k_coef=1):
    """
    :param return_embeddings: whether to return intermediat features -- for TLA
    :param batch_norm: whether utilize BatchNormalization, or Dropout -- for GloroNet use Dropout, 
    since BatchNorm modifies the Lipschitz constant
    """

    inpt = tf.keras.layers.Input(shape=input_shape)

    ###
    hdn = SpectralConv2D(filters=64, kernel_size=3, padding='same', k_coef_lip=k_coef)(inpt)
    hdn = tf.keras.activations.relu(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.1)(hdn)

    hdn = SpectralConv2D(filters=64, kernel_size=3, padding='same', k_coef_lip=k_coef)(hdn)
    hdn = tf.keras.activations.relu(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.1)(hdn)

    hdn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(hdn)
    ###

    ###
    hdn = SpectralConv2D(filters=128, kernel_size=3, padding='same', k_coef_lip=k_coef)(hdn)
    hdn = tf.keras.activations.relu(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.2)(hdn)

    hdn = SpectralConv2D(filters=128, kernel_size=3, padding='same', k_coef_lip=k_coef)(hdn)
    hdn = tf.keras.activations.relu(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.2)(hdn)

    hdn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(hdn)
    ###

    ###
    hdn = SpectralConv2D(filters=256, kernel_size=3, padding='same', k_coef_lip=k_coef)(hdn)
    hdn = tf.keras.activations.relu(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.3)(hdn)

    hdn = SpectralConv2D(filters=256, kernel_size=3, padding='same', k_coef_lip=k_coef)(hdn)
    hdn = tf.keras.activations.relu(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.3)(hdn)

    hdn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(hdn)
    ###

    hdn = tf.keras.layers.Flatten()(hdn)

    hdn = SpectralDense(256, k_coef_lip=k_coef)(hdn)
    hdn = tf.keras.activations.relu(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.4)(hdn)

    embeddings = SpectralDense(256, k_coef_lip=k_coef)(hdn)
    hdn = tf.keras.activations.relu(embeddings)
    hdn = tf.keras.layers.Dropout(rate=0.3)(hdn)

    out = SpectralDense(n_classes, k_coef_lip=k_coef)(hdn)

    if return_embeddings:
        model = tf.keras.models.Model(inputs=inpt, outputs=[out, embeddings])
    else:
        model = tf.keras.models.Model(inputs=inpt, outputs=out)
    return model


def model_AT_mnist_deelLip(input_shape, n_classes, k_coef):
    """Do not use BatchNormalization -- it changes the Lip constant since it has scaling
    Use Dropout instead."""
    inpt = tf.keras.layers.Input(shape=input_shape)

    hdn = SpectralConv2D(filters=16, kernel_size=3, padding='same', k_coef_lip=k_coef)(inpt)
    hdn = tf.keras.activations.relu(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.3)(hdn)

    hdn = SpectralConv2D(filters=32, kernel_size=3, padding='same', k_coef_lip=k_coef)(hdn)
    hdn = tf.keras.activations.relu(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.3)(hdn)

    hdn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(hdn)

    hdn = SpectralConv2D(filters=64, kernel_size=3, padding='same', k_coef_lip=k_coef)(hdn)
    hdn = tf.keras.activations.relu(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.3)(hdn)

    hdn = SpectralConv2D(filters=64, kernel_size=3, padding='same', k_coef_lip=k_coef)(hdn)
    hdn = tf.keras.activations.relu(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.3)(hdn)

    hdn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(hdn)

    hdn = tf.keras.layers.Flatten()(hdn)

    hdn = SpectralDense(256, k_coef_lip=k_coef)(hdn)
    hdn = tf.keras.activations.relu(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.3)(hdn)

    hdn = SpectralDense(128, k_coef_lip=k_coef)(hdn)
    hdn = tf.keras.activations.relu(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.3)(hdn)

    out = SpectralDense(n_classes, k_coef_lip=k_coef)(hdn)

    model = tf.keras.models.Model(inputs=inpt, outputs=out)

    return model


def configurable_model_ensemble_mnist(
        n_filters = [],
        n_neurons = [],
        input_shape=(28, 28, 1), 
        n_classes=10
        ):
    """
    Defines custom model for creating ensemble peer. Uses only 3x3 convolutions, with padding='same' and ReLU actiavtions

    n_filters: list of integer values, representing the number of filters for the conv layer. If -1 value is found, this indicates 
    a MaxPooling layer. 
    """

    assert len(n_filters) > 0, ValueError("No filters given.")

    inpt = tf.keras.layers.Input(shape=input_shape)

    hdn = inpt
    for i in range(len(n_filters)):
        if n_filters[i] == -1:
            hdn = tf.keras.layers.MaxPooling2D(pool_size=2)(hdn)
        else:
            hdn = tf.keras.layers.Conv2D(filters=n_filters[i], kernel_size=3, padding='same')(hdn)
            hdn = tf.keras.layers.BatchNormalization()(hdn)
            hdn = tf.keras.activations.relu(hdn)

    hdn = tf.keras.layers.Flatten()(hdn)

    for i in range(len(n_neurons)):
        hdn = tf.keras.layers.Dense(n_neurons[i])(hdn)
        hdn = tf.keras.layers.BatchNormalization()(hdn)
        hdn = tf.keras.activations.relu(hdn)

    out = tf.keras.layers.Dense(n_classes, activation=None)(hdn)
    model = tf.keras.models.Model(inputs=inpt, outputs=out)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def model_AT_mnist_quantized(input_shape, n_classes, w_bits, a_bits):
    inpt = tf.keras.layers.Input(shape=input_shape)

    hdn = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same',
                                kernel_constraint=QuantizeWeights(w_bits),
                                bias_constraint=QuantizeWeights(w_bits),
                                )(inpt)
    # hdn = quantizedReLU(hdn, a_bits)
    hdn = tf.math.tanh(hdn)

    hdn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(hdn)

    hdn = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',
                                kernel_constraint=QuantizeWeights(w_bits),
                                bias_constraint=QuantizeWeights(w_bits),
                                )(hdn)
    # hdn = quantizedReLU(hdn, a_bits)
    hdn = tf.math.tanh(hdn)

    hdn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(hdn)

    hdn = tf.keras.layers.Flatten()(hdn)

    hdn = tf.keras.layers.Dense(256,
                                kernel_constraint=QuantizeWeights(w_bits),
                                bias_constraint=QuantizeWeights(w_bits),
                                )(hdn)
    hdn = tf.math.tanh(hdn)

    hdn = tf.keras.layers.Dense(256,
                                kernel_constraint=QuantizeWeights(w_bits),
                                bias_constraint=QuantizeWeights(w_bits),
                                )(hdn)
    hdn = tf.math.tanh(hdn)

    out = tf.keras.layers.Dense(10,
                                kernel_constraint=QuantizeWeights(w_bits),
                                bias_constraint=QuantizeWeights(w_bits),
                                )(hdn)

    model = tf.keras.models.Model(inputs=inpt, outputs=out)
    return model


def identity_block(x, filters, dropout=False, deel=False):
    x_skip = x

    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
    if deel:
        x = tf.keras.layers.Dropout(rate=0.3)(x)
    else:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    if dropout and not deel:
        x = tf.keras.layers.Dropout(rate=0.2)(x)

    x = tf.keras.layers.Conv2D(filters, (3,3), padding='same')(x)
    if deel:
        x = tf.keras.layers.Dropout(rate=0.3)(x)
    else:
        x = tf.keras.layers.BatchNormalization(axis=3)(x)

    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.activations.relu(x)

    return x 


def conv_block(x, filters, dropout=False, deel=False):
    x_skip = x
    
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', strides=2)(x)
    if deel:
        x = tf.keras.layers.Dropout(rate=0.3)(x)
    else:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    if dropout and not deel:
        x = tf.keras.layers.Dropout(rate=0.2)(x)

    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
    if deel:
        x = tf.keras.layers.Dropout(rate=0.3)(x)
    else:
        x = tf.keras.layers.BatchNormalization()(x)

    x_skip = tf.keras.layers.Conv2D(filters, 1, strides=2)(x_skip)

    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.activations.relu(x)

    return x


def res_net9(input_shape=(64, 64, 3), n_classes=10, return_embeddings=False, deel=False):
    """
    Generic ResNet-N model. Inspired from https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/
    """
    filter_size = 16

    inpt = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filter_size, kernel_size=3, padding='same')(inpt)
 
    block_layers = [2, 2, 2, 2]

    for i in range(4):
        if i == 0:
            for _ in range(block_layers[i]):
                x = identity_block(x, filter_size, deel=deel)
        else:
            filter_size *= 2
            x = conv_block(x, filter_size, deel=deel)
            for _ in range(block_layers[i] - 1):
                x = identity_block(x, filter_size, dropout=True, deel=deel)

    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)

    embedd = tf.keras.layers.Dense(256)(x)
    x = tf.keras.activations.relu(embedd)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    out = tf.keras.layers.Dense(n_classes)(x)

    if return_embeddings:
        model = tf.keras.models.Model(inputs=inpt, outputs=[out, embedd])
    else:
        model = tf.keras.models.Model(inputs=inpt, outputs=out)

    return model


def celeba_model(input_shape, n_classes, return_embeddings=False):

    inpt = tf.keras.layers.Input(shape=input_shape)

    hdn = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')(inpt)
    hdn = tf.keras.activations.relu(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.1)(hdn)
    hdn = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2, padding='same')(hdn)
    
    hdn = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(hdn)
    hdn = tf.keras.activations.relu(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.2)(hdn)
    hdn = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(hdn)

    hdn = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(hdn)
    hdn = tf.keras.activations.relu(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.2)(hdn)
    hdn = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(hdn)

    hdn = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same')(hdn)
    hdn = tf.keras.activations.relu(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.3)(hdn)
    hdn = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(hdn)

    hdn = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same')(hdn)
    hdn = tf.keras.activations.relu(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.3)(hdn)
    hdn = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(hdn)

    embedd = tf.keras.layers.GlobalMaxPool2D()(hdn)
    hdn = tf.keras.layers.Flatten()(embedd)

    hdn = tf.keras.layers.Dense(n_classes)(hdn)

    if return_embeddings:
        model = tf.keras.models.Model(inputs=inpt, outputs=[hdn, embedd])
    else:
        model = tf.keras.models.Model(inputs=inpt, outputs=hdn)
    
    return model
