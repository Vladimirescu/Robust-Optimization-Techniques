from art.attacks.evasion import AdversarialPatchTensorFlowV2
from art.estimators.classification import TensorFlowV2Classifier
from plot_adv_samples import *


def train_step(model, images, labels):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


if __name__ == "__main__":

    dataset_type = 6 # also used to select the list of paths to load
    idx_methods = [0, 1, 2, 3, 4, 5, 6, 7] # used to select which methods from names list will be plotted 
    generate_from_first_n = 500
    class_id = 21
    target_id = 61

    paths = datasets_paths[dataset_type - 1]
    paths = [paths[idx] for idx in idx_methods]
    save_path = f"adversarial_patches_results/{datasets[dataset_type]}/class_{class_id}_target_{target_id}/"

    x_train, y_train, x_test, y_test, input_shape, n_classes = load_dataset(dataset_type)

    models = []
    names = [names[idx] for idx in idx_methods]

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
        elif dataset_type == 6:
            model = res_net9(input_shape, n_classes, filters_start=8, deel=True)
        else:
            raise ValueError("Not implemented yet.")
        
        model.load_weights(path)
        models.append(model)

    x_train_class_id = x_train[y_train == class_id]
    x_train_class_id = x_train_class_id[:generate_from_first_n]

    for i, model in enumerate(models):

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ['accuracy']
        model.trainable = False
        model.compile(optimizer, loss, metrics)
        
        art_classifier = TensorFlowV2Classifier(
            model=model,
            loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            train_step=train_step,
            nb_classes=n_classes,
            input_shape=input_shape,
            clip_values=(-1, 1),
        )

        apt = AdversarialPatchTensorFlowV2(classifier=art_classifier, batch_size=256, max_iter=100,
                                           patch_shape=(64, 64, 3))

        target = [target_id] * x_train_class_id.shape[0]
        patch, circular_patch = apt.generate(x_train_class_id, target)

        print(patch.shape, circular_patch.shape)

        save_path_model = save_path + "/" + names[i] + "/"
        if not os.path.exists(save_path_model):
            os.makedirs(save_path_model)

        np.save(save_path_model + "patch.npy", patch)
        np.save(save_path_model + "circular_patch.npy", circular_patch)

        plt.figure()
        plt.imshow((patch * 255).astype(np.uint8))
        plt.axis("off")
        plt.savefig(save_path_model + "advpatch.png", bbox_inches="tight", dpi=300, pad_inches=0.01, orientation='landscape', transparent=True)
        plt.close()

        plt.figure()
        plt.imshow((circular_patch * 255).astype(np.uint8))
        plt.axis("off")
        plt.savefig(save_path_model + "advpatch_circular.png", bbox_inches="tight", dpi=300, pad_inches=0.01, orientation='landscape', transparent=True)
        plt.close()



