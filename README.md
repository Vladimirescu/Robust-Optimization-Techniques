# Robust-Optimization-Techniques
This repo contains [Tensorflow 2.+](https://www.tensorflow.org/) implementations for robust training algorithms used in image classification problems.

## Training scripts
``train_AT.py`` - Adversarial Training, Ensemble Adversarial Training, and Baseline model training

``train_TRADES.py`` - regularization training using [TRADES](http://proceedings.mlr.press/v97/zhang19p/zhang19p.pdf)

``train_DeepDefense.py`` - regularization training using [DeepDefense](https://proceedings.neurips.cc/paper/2018/hash/8f121ce07d74717e0b1f21d122e04521-Abstract.html)

``train_TLA.py`` - regularization training using [Metric Learning (TLA)](https://proceedings.neurips.cc/paper/2019/hash/c24cd76e1ce41366a4bbe8a49b02a028-Abstract.html)

``train_RS.py`` - training with Gaussian augmentation: [randomized Smoothing](https://proceedings.mlr.press/v97/cohen19c.html)

``train_MACER.py`` - certified robustness training using [MACER](https://arxiv.org/abs/2001.02378)

``train_DeelLip.py`` - certified robustness training using [DeelLip](https://openaccess.thecvf.com/content/CVPR2021/html/Serrurier_Achieving_Robustness_in_Classification_Using_Optimal_Transport_With_Hinge_Regularization_CVPR_2021_paper.html) + 2 modifications. Official DeelLip library [here](https://github.com/deel-ai/deel-lip).

``train_GloroNet.py`` - certified robustness training using [GloroNet](https://arxiv.org/pdf/2102.08452.pdf). Official GloroNet library [here](https://github.com/klasleino/gloro).

``train_LMT.py`` -- certified robustness training using Lipschitz Margin Training [LMT](https://proceedings.neurips.cc/paper/2018/hash/485843481a7edacbfce101ecb1e4d2a8-Abstract.html).

## Trained model files 
The trained model file for a dataset, method, and a specific configuration for that method, can be accessed from:
```
models_{method}  
│
└───{dataset}
│   │
│   └───parameter1={value1}_parameter2={value2}_ ...
│       │
         --   model.hdf5

## Testing/Plotting scripts

``plot_acc_vs_eps.py`` - create & save accuracy vs. perturbation plot, for a given attack, for a set of models

``plot_adv_samples.py`` - attack and plot adversarial examples, for a given attack, for a set of models

``plot_perturb_hist.py`` - create & save histograms for the magnitude of perturbation, for a given attack, for a set of models

``plot_TLA_embeddings.py`` - apply TSNE on internal representations for a set of models trained with TLA (or other methods), obtained from positive, negative, and targeted adversarial samples generated from positive inputs 

``test_external_images.py`` -- apply models on external images, provided by the user (used for Adversarial Patch testing)

## Others

``models.py`` - contains neural network architectures used for MNIST, FMNIST, CIFAR-10, CelebA, EuroSAT-RGB and BelgiumTS datasets

``utils.py`` - contains functionalities for loading a dataset, and ``tf.keras.callback.Callback`` for training a model with different Lipschitz constraints

``AT_utils.py`` - contains functionalities for generating adversarial samples, adversarial training, and ensemble adversarial training

``make_adv_patch.py`` - generates targeted/untargeted [Adversarial Patch](https://arxiv.org/pdf/1712.09665.pdf) for a given set of images, using the [ART Library](https://github.com/Trusted-AI/adversarial-robustness-toolbox).

