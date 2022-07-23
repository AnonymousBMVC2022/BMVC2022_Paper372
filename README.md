## ASCL: Adaptive Soft Contrastive Learning

<p align="center">
    <img src="./sources/method.png" alt="drawing" width="800"/>
</p>

This is the official PyTorch implementation of BMVC2022 submission 372: SSR: An Efficient and Robust Framework for Learning with Noisy Labels. 

### Abstract
Despite the large progress in supervised learning with neural networks, there are significant challenges in obtaining high-quality, large-scale and accurately labeled datasets. In such a context, how to learn in the presence of noisy labels has received more and more attention. As a relatively comprehensive problem, in order to achieve good results, the current methods often integrate technologies from multiple fields, such as supervised learning, semi-supervised learning, transfer learning, and so on. At the same time, these methods often make strong or weak assumptions about the noise of the data. This also inevitably brings about the problem of model robustness.
Striving for simplicity and robustness, we propose an efficient and robust framework named Sample Selection and Relabelling(SSR), that minimizes the number of modules and hyperparameters required, and that achieves good results in various conditions. In the heart of our method is a sample selection and relabelling mechanism based on a non-parametric KNN classifier $g_q$ and a parametric model classifier $g_p$, respectively, to select the clean samples and gradually relabel the closed-set noise samples.
Without bells and whistles, such as model co-training, self-supervised pertaining, and semi-supervised learning, and with robustness concerning settings of its few hyper-parameters, our method significantly surpasses previous methods on both CIFAR10/CIFAR100 with synthetic noise and real-world noisy datasets such as WebVision, Clothing1M and ANIMAL-10N. Code is available at https://github.com/AnnoymousRepo/BMVC2022.

[//]: # (```)

[//]: # (@inproceedings{)

[//]: # (    chen2022icpr,)

[//]: # (    title={ASCL: Adaptive Soft Contrastive Learning},)

[//]: # (    author={Chen Feng and Ioannis Patras},)

[//]: # (    booktitle={International Conference on Pattern Recognition},)

[//]: # (    year={2022},)

[//]: # (})

[//]: # (```)

### Preparation
- pytorch
- tqdm
- wandb

### Usage
An example run on CIFAR100 dataset with 90% symmetric noise:
```
python main.py --gpuid 0 --run_path 0.3sym_0.5open_fc_balanced_r0.8_s1.0_optionalfc_hardrelabelling_cosine_cifar10_nowarmup_weakeval --noise_ratio 0.3 --open_ratio 0.5 --balanced_sampler --theta_r 0.95 --theta_s 1.0 --noise_mode sym --warmup_epochs 0
```

For users who are not familiar with wandb, please try main_simple.py.
```
python main_simple.py  --epochs 200 --gpuid 0 --dataset cifar100 --data_path data/CIFAR100 --K 5  --type ascl --t1 0.1 --t2 0.05 --aug weak_augment
```


For experiments on ImageNet-1K, we simply modified the official MoCo repo. 
For brevity of this repo, we provide a modified `builder.py` which can be utilized with the original [MoCo](https://github.com/facebookresearch/moco) repo.


### License
This project is licensed under the terms of the MIT license.
