from plot_adv_samples import *
import imageio


if __name__ == "__main__":

    dataset_type = 6 # also used to select the list of paths to load
    idx_methods = [0, 1, 2, 3, 5, 6, 7]  # used to select which methods from names list will be plotted 

    image_files = [
        "external_images_v2/park_proh/MicrosoftTeams-image (61).png",
        "external_images_v2/park_proh_v3/MicrosoftTeams-image (60).png",
    ]
    check_class = 41

    _, _, _, _, input_shape, n_classes = load_dataset(dataset_type)

    models = []
    images = []
    paths = datasets_paths[dataset_type - 1]
    paths = [paths[idx] for idx in idx_methods]
    names = [names[idx] for idx in idx_methods]

    if dataset_type == 6:
        # process images
        input_shape = (96, 96, 3)

        for file in image_files:
            x = imageio.imread(file)
            if x.shape[-1] == 4:
                x = x[..., :3]
            if x.shape[0] != input_shape[0] or x.shape[1] != input_shape[1]:
                x = tf.image.resize(x, [input_shape[0], input_shape[1]]).numpy()
            images.append((x - 127.5) / 127.5)

            if not os.path.exists(f"{file}_resized.png"):
                plt.figure()
                plt.imshow(x.astype(np.uint8))
                plt.axis("off")
                plt.savefig(f"{file}_resized.png", bbox_inches="tight", dpi=300, pad_inches=0.01)
    else:
        raise ValueError("Not implemented.")

    if dataset_type in [1, 2]:
        model_base = model_AT_mnist(input_shape, n_classes, batch_norm=False)
        if dataset_type == 1:
            model_base.load_weights("models_AT/baseline/MNIST/model.hdf5")
        else:
            model_base.load_weights("models_AT/baseline/FMNIST/model.hdf5")
    elif dataset_type == 3:
        model_base = model_AT_cifar(input_shape, n_classes, batch_norm=False)
        model_base.load_weights("models_AT/baseline/CIFAR-10/model.hdf5")
    elif dataset_type == 4:
        model_base = res_net9(input_shape, n_classes)
        model_base.load_weights("models_AT/baseline/eurosat/model.hdf5")
    elif dataset_type == 5:
        model_base = celeba_model(input_shape, n_classes)
        model_base.load_weights("models_AT/baseline/CelebA/model.hdf5")
    elif dataset_type == 6:
        model_base = res_net9(input_shape=input_shape, n_classes=n_classes, filters_start=8, deel=True)
        model_base.load_weights("models_AT/baseline/BelgiumTS/model.hdf5")
    else:
        raise ValueError("Not implemented yet.")
    
    names = ["Baseline"] + names
    models.append(model_base)

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

    for i, model in enumerate(models):

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ['accuracy']
        model.trainable = False
        model.compile(optimizer, loss, metrics)

        print(names[i])
        for j, im in enumerate(images):
            logits = model(im[np.newaxis, ...]).numpy()
            if names[i] == "MACER":
                print(logits)
            pred = np.argsort(logits[0])[::-1]
            logits = logits[0][pred]
            prob = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-7)
            print(image_files[j], "top 2 predictions: ", pred[:2], prob[:2], "correct class prob: ", prob[pred == check_class])