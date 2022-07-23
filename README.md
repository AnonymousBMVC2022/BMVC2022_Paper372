## ASCL: Adaptive Soft Contrastive Learning

<p align="center">
    <img src="./sources/method.png" alt="drawing" width="800"/>
</p>

This is the official PyTorch implementation of BMVC2022 submission 372: SSR: An Efficient and Robust Framework for Learning with Noisy Labels. 

### Abstract
Despite the large progress in supervised learning with neural networks, there are significant challenges in obtaining high-quality, large-scale and accurately labeled datasets. In such a context, how to learn in the presence of noisy labels has received more and more attention. As a relatively comprehensive problem, in order to achieve good results, the current methods often integrate technologies from multiple fields, such as supervised learning, semi-supervised learning, transfer learning, and so on. At the same time, these methods often make strong or weak assumptions about the noise of the data. This also inevitably brings about the problem of model robustness.
Striving for simplicity and robustness, we propose an efficient and robust framework named Sample Selection and Relabelling(SSR), that minimizes the number of modules and hyperparameters required, and that achieves good results in various conditions. In the heart of our method is a sample selection and relabelling mechanism based on a non-parametric KNN classifier $g_q$ and a parametric model classifier $g_p$, respectively, to select the clean samples and gradually relabel the closed-set noise samples.
Without bells and whistles, such as model co-training, self-supervised pertaining, and semi-supervised learning, and with robustness concerning settings of its few hyper-parameters, our method significantly surpasses previous methods on both CIFAR10/CIFAR100 with synthetic noise and real-world noisy datasets such as WebVision, Clothing1M and ANIMAL-10N. Code is available at https://github.com/AnnoymousRepo/BMVC2022.

### Preparation
- pytorch
- tqdm
- wandb

### Usage
Example runs on CIFAR100 dataset with symmetric noise:
```
python main_cifar.py --theta_r 0.8 --noise_mode sym --noise_ratio 0.9 --dataset cifar100 --dataset_path path_to_cifar100

python main_cifar.py --theta_r 0.8 --noise_mode sym --noise_ratio 0.8 --dataset cifar100 --dataset_path path_to_cifar100

python main_cifar.py --theta_r 0.8 --noise_mode sym --noise_ratio 0.5 --dataset cifar100 --dataset_path path_to_cifar100

python main_cifar.py --theta_r 0.9 --noise_mode sym --noise_ratio 0.2 --dataset cifar100 --dataset_path path_to_cifar100
```

For users who are not familiar with wandb, please try main_cifar_simple.py with same config.

### License
This project is licensed under the terms of the MIT license.
