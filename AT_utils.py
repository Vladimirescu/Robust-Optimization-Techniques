"""
Implements all the functionalities used for adversarial training algorithms.
"""
import foolbox as fb
import tensorflow as tf
import numpy as np
# import wandb
# from wandb.integration.keras import WandbCallback
from models import configurable_model_ensemble_mnist


at_attack_types = {
    "pgd": {
        "steps_per_batch": 15,
        "epsilon": {
            "MNIST": 0.5,
            "FMNIST": 0.5,
            "CIFAR-10": 0.3,
            "eurosat": 0.1
        }
    },
    "fgsm": {
        "steps_per_batch": None,
        "epsilon": {
            "MNIST": 2.0,
            "FMNIST": 2.0,
            "CIFAR-10": 1.0,
            "eurosat": 0.1,
            "CelebA": 0.1
        }
    },
    "ddn": {
        "steps_per_batch": 15,
        "epsilon": {
            "MNIST": 0.5,
            "FMNIST": 0.5,
            "CIFAR-10": 0.3,
            "eurosat": 0.1
        }
    }
}

ensemble_configs = {
    0: {
        "n_filters": [8, -1, 16, -1, 32, -1],
        "n_neurons": [128]
    },
    1: {
        "n_filters": [64, -1, -1],
        "n_neurons": [256, 128, 64]
    },
    2: {
        "n_filters": [16, -1, 32, -1, 64, -1],
        "n_neurons": [32]
    },
    3: {
        "n_filters": [64, -1, 128, -1],
        "n_neurons": [64, 32, 16]
    },
    4: {
        "n_filters": [64, -1, 128, -1],
        "n_neurons": [32, 16]
    },
}


def make_adv_batch(foolbox_model, 
                   attacker, 
                   epsilon, x, y, batch_size=4096):
    x = tf.cast(x, float)
    y = tf.cast(y, tf.int32)
    
    if isinstance(epsilon, list):
        clipped_adv = np.empty((len(epsilon), *x.shape))
        is_adv = np.empty((len(epsilon), int(x.shape[0])))
    else:
        clipped_adv = np.empty_like(x)
        is_adv = np.empty(x.shape[0])

    for i in range(x.shape[0] // batch_size):
        raw, clipped_i, is_adv_i = attacker(foolbox_model, 
                                            x[i*batch_size: (i+1)*batch_size], 
                                            y[i*batch_size: (i+1)*batch_size], 
                                            epsilons=epsilon)
        if isinstance(epsilon, list):
            clipped_adv[:, i*batch_size: (i+1)*batch_size, ...] = clipped_i
            is_adv[:, i*batch_size: (i+1)*batch_size] = is_adv_i
        else:
            clipped_adv[i*batch_size: (i+1)*batch_size] = clipped_i
            is_adv[i*batch_size: (i+1)*batch_size] = is_adv_i

    raw, clipped_i, is_adv_i = attacker(foolbox_model, 
                                        x[-(x.shape[0] % batch_size):], 
                                        y[-(x.shape[0] % batch_size):], 
                                        epsilons=epsilon)
    if isinstance(epsilon, list):
        clipped_adv[:, -(x.shape[0] % batch_size):] = clipped_i
        is_adv[:, -(x.shape[0] % batch_size):] = is_adv_i
    else:
        clipped_adv[-(x.shape[0] % batch_size):] = clipped_i
        is_adv[-(x.shape[0] % batch_size):] = is_adv_i

    return raw, clipped_adv, is_adv



def adversarial_training_simple(dset_name,
                                model, 
                                x_train, 
                                y_train, 
                                x_test, 
                                y_test, 
                                epochs, 
                                batch_size, 
                                attack,
                                save_names,
                                log_metrics=False,
                                full_adversarial=True):
    """
    Function used for adversarial training. After each epoch, a number of adversarial examples are generated
    and the model is either trained solely on them or the current training set is extended. 

    full_adversarial -- whether the model is trained only on adversarial samples or not 

    dset_name: name of dataset the model is trained on | attacks have different config for each dataset
    save_names: tuple containing the base folder and the name of the saved model 
    """
    x_train, x_test = tf.cast(x_train, float), tf.cast(x_test, float)
    y_train, y_test = tf.cast(y_train, tf.int32), tf.cast(y_test, tf.int32)

    model_save_path, filename = save_names

    epsilon = at_attack_types[attack]["epsilon"][dset_name]
    steps = at_attack_types[attack]["steps_per_batch"]
    save_path = model_save_path + "/" + filename + f"_epsilon={epsilon}.hdf5"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_loss',
                                            verbose=1, save_best_only=True, save_weights_only=True)
    
    if attack == "ddn":
        attacker = fb.attacks.DDNAttack(steps=steps)
    elif attack == "pgd":
        attacker = fb.attacks.L2ProjectedGradientDescentAttack(steps=steps)
    elif attack == "fgsm":
        attacker = fb.attacks.L2FastGradientAttack()

    callback_list = [checkpoint]
    
    if log_metrics:
        wandb.init(
            project="disert"
        )
        wandb_callback = WandbCallback(monitor="loss", log_weights=True, log_gradients=True, training_data=(x_train, y_train), 
                            validation_data=(x_test, y_test), save_model=False)
        callback_list.append(wandb_callback)

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=512) # train for 1 epoch
    for i in range(epochs - 1):
        model.trainable = False
        foolbox_model = fb.models.TensorFlowModel(model, bounds=(-1, 1))
        model.trainable = True
        """Generate adv example for each image from x_train -- use batches, can't compute gradients for all images at once"""
        raw, clipped_adv, is_adv = make_adv_batch(foolbox_model, attacker, epsilon, x_train, y_train, batch_size=1024)

        """Create extended training dataset and shuffle it"""
        if full_adversarial:
            x_train_extended = clipped_adv
            y_train_extended = y_train.numpy()
        else:
            x_train_extended = np.concatenate([x_train, clipped_adv], axis=0)
            y_train_extended = np.concatenate([y_train, y_train], axis=0)

        idx = np.random.permutation(x_train_extended.shape[0])

        x_train_extended = x_train_extended[idx]
        y_train_extended = y_train_extended[idx]

        """Fit for 1 epoch"""
        model.fit(x_train_extended, y_train_extended, validation_data=(x_test, y_test), epochs=1, batch_size=batch_size,
                  callbacks=callback_list)
        
    
def ensemble_adversarial_training(dset_name,
                                  model,
                                  x_train, 
                                  y_train,
                                  x_test,
                                  y_test,
                                  epochs,
                                  epochs_ensemble,
                                  batch_size,
                                  n_ensembles,
                                  attack,
                                  save_names,
                                  log_metrics=False):
    """
    Ensemble adversarial trainig. A number of models, with given configs, are trained, 
    """
    
    x_train, x_test = tf.cast(x_train, float), tf.cast(x_test, float)
    y_train, y_test = tf.cast(y_train, tf.int32), tf.cast(y_test, tf.int32)

    model_save_path, filename = save_names

    epsilon = at_attack_types[attack]["epsilon"][dset_name]
    steps = at_attack_types[attack]["steps_per_batch"]
    save_path = model_save_path + "/" + filename + f"_epsilon={epsilon}.hdf5"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_loss',
                                            verbose=1, save_best_only=True, save_weights_only=True)
    
    if attack == "ddn":
        attacker = fb.attacks.DDNAttack(steps=steps)
    elif attack == "pgd":
        attacker = fb.attacks.L2ProjectedGradientDescentAttack(steps=steps)
    elif attack == "fgsm":
        attacker = fb.attacks.L2FastGradientAttack()

    callback_list = [checkpoint]
    
    if log_metrics:
        wandb.init(
            project="disert"
        )
        wandb_callback = WandbCallback(monitor="loss", log_weights=True, log_gradients=True, training_data=(x_train, y_train), 
                            validation_data=(x_test, y_test), save_model=False)
        callback_list.append(wandb_callback)

    ensemble_models = []
    adversarials_from_ensemble = []
    assert n_ensembles <= 5, ValueError("Number of ensembles should be limited to 5.")
    for i in range(n_ensembles):
        n_filters = ensemble_configs[i]["n_filters"]
        n_neurons = ensemble_configs[i]["n_neurons"]

        if dset_name in ["MNIST", "FMNIST"]:
            ens_model_i = configurable_model_ensemble_mnist(n_filters, n_neurons)
        elif dset_name == "CIFAR-10":
            ens_model_i = configurable_model_ensemble_mnist(n_filters, n_neurons, input_shape=(32, 32, 3))

        ens_model_i.fit(x_train, y_train, epochs=epochs_ensemble, batch_size=batch_size, verbose=0)
        res = ens_model_i.evaluate(x_test, y_test, verbose=0)
        res = ens_model_i.evaluate(x_test, y_test, verbose=0)
        ensemble_models.append(ens_model_i)
    
        print(f"\nTrained ensemble model {i} achieved val_accuracy={res[1]}.\n")

        """Compute adversarial examples for the trained model from ensemble"""
        foolbox_model = fb.models.TensorFlowModel(ens_model_i, bounds=(-1, 1))
        raw, clipped_adv, is_adv = make_adv_batch(foolbox_model, attacker, epsilon, x_train, y_train, batch_size=4096)

        adversarials_from_ensemble.append(clipped_adv)

    """
    Train model on adversarial examples generated from ensembles and from itself.
    Ensemble models are are alternated at each epoch for generating adversarial samples. 
    """

    for i in range(epochs):
        idx_ensemble = (i % n_ensembles)
        adv_samples = adversarials_from_ensemble[idx_ensemble]

        perm = np.random.permutation(adv_samples.shape[0])
        
        adv_samples_i = adv_samples[perm]
        y_train_i = y_train.numpy()[perm]

        """Fit for 1 epoch"""
        """Save the model and metrics only when the model has already passed through all adversarials, generated by all ensemble networks,\
        at least three times -- ensure the model adapts to most of the examples first"""
        callback_list_i = callback_list if i > 3 * n_ensembles else []
        model.fit(
            adv_samples_i, y_train_i, validation_data=(x_test, y_test), epochs=1, batch_size=batch_size,
                  callbacks=callback_list_i
        )


