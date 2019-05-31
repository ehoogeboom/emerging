# Emerging Convolutions for Generative Flows

Code to reproduce results in [paper](https://arxiv.org/abs/1901.11137) [blog](https://ehoogeboom.github.io/post/invertible_convs/)

If you use our work, please cite us: 
```
Emiel Hoogeboom, Rianne van den Berg, and Max Welling. Emerging Convolutions for Generative Normalizing Flows. International Conference on Machine Learning, 2019.
```

A BibTeX entry for LaTeX users is:
```
@inproceedings{
hoogeboom2019emerging,
title={Emerging Convolutions for Generative Normalizing Flows},
author={Emiel Hoogeboom and Rianne van den Berg and Max Welling},
booktitle={International conference on machine learning},
year={2019},
url={https://arxiv.org/abs/1901.11137},
}
```

The source is adapted from [Glow: Generative Flow with Invertible 1x1 Convolutions](https://github.com/openai/glow)

## Requirements
- Horovod (tested with 0.15.2)
- Tensorflow (tested with 1.12)

## Download datasets
CIFAR10 is automatically downloaded.
Galaxy images need to be downloaded [here](https://github.com/SpaceML/merger_transfer_learning).

ImageNet 32x32 and 64x64 was downloaded from the link on the Glow github: `https://storage.googleapis.com/glow-demo/data/{dataset_name}-tfr.tar'
with `imagenet-oord` as dataset_name. 


##### Galaxy images results

Periodic:
```
mpiexec -n 4 python train.py --problem space --image_size 32 --n_level 2 --depth 8 --flow_permutation 5 --flow_coupling 1 --seed 2 --lr 0.001 --n_bits_x 8 --epochs 6001
```

Emerging:
```
mpiexec -n 4 python train.py --problem space --image_size 32 --n_level 2 --depth 8 --flow_permutation 3 --flow_coupling 1 --seed 2 --lr 0.001 --n_bits_x 8 --epochs 6001
```

Baseline (Glow):
```
mpiexec -n 4 python train.py --problem space --image_size 32 --n_level 2 --depth 8 --flow_permutation 2 --flow_coupling 1 --seed 2 --lr 0.001 --n_bits_x 8 --epochs 6001
```

##### CIFAR-10 results

Emerging:
```
mpiexec -n 4 python train.py --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation 3 --flow_coupling 1 --seed 2 --learnprior --lr 0.001 --n_bits_x 8 --epochs 4001
```

Baseline (Glow):
```
mpiexec -n 4 python train.py --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation 2 --flow_coupling 1 --seed 2 --learnprior --lr 0.001 --n_bits_x 8 --epochs 4001
```


##### CIFAR-10 results (smaller architectures)

Replace ? with either 8 or 4, depending on the experiment.

Emerging:
```
mpiexec -n 4 python train.py --problem cifar10 --image_size 32 --n_level 3 --depth ? --flow_permutation 3 --flow_coupling 1 --seed 2 --learnprior --lr 0.001 --n_bits_x 8 --epochs 4001
```

Baseline (Glow):
```
mpiexec -n 4 python train.py --problem cifar10 --image_size 32 --n_level 3 --depth ? --flow_permutation 2 --flow_coupling 1 --seed 2 --learnprior --lr 0.001 --n_bits_x 8 --epochs 4001
```

##### ImageNet 32x32 results

Emerging:
```
mpiexec -n 4 python train.py --problem imagenet-oord --image_size 32 --n_level 3 --depth 48 --flow_permutation 3 --flow_coupling 1 --seed 0 --learnprior --lr 0.001 --n_bits_x 8
```

Baseline (Glow):
```
mpiexec -n 4 python train.py --problem imagenet-oord --image_size 32 --n_level 3 --depth 48 --flow_permutation 2 --flow_coupling 1 --seed 0 --learnprior --lr 0.001 --n_bits_x 8
```


##### ImageNet 64x64 results
Emerging:
```
mpiexec -n 4 python train.py --problem imagenet-oord --image_size 64 --n_level 4 --depth 48 --flow_permutation 3 --flow_coupling 1 --seed 0 --learnprior --lr 0.001 --n_bits_x 8
```

Baseline (Glow):
```
mpiexec -n 4 python train.py --problem imagenet-oord --image_size 64 --n_level 4 --depth 48 --flow_permutation 2 --flow_coupling 1 --seed 0 --learnprior --lr 0.001 --n_bits_x 8
```


##### 1x1 Convolution results
QR 1x1:
```
mpiexec -n 4 python train.py --problem cifar10 --image_size 32 --n_level 3 --depth 8 --flow_permutation 2 --flow_coupling 1 --seed 2 --learnprior --lr 0.001 --n_bits_x 8 --epochs 3501 --decomposition QR
```

PLU 1x1 (Glow):
```
mpiexec -n 4 python train.py --problem cifar10 --image_size 32 --n_level 3 --depth 8 --flow_permutation 2 --flow_coupling 1 --seed 2 --learnprior --lr 0.001 --n_bits_x 8 --epochs 3501 --decomposition PLU
```

Baseline 1x1 (Glow):
```
mpiexec -n 4 python train.py --problem cifar10 --image_size 32 --n_level 3 --depth 8 --flow_permutation 2 --flow_coupling 1 --seed 2 --learnprior --lr 0.001 --n_bits_x 8 --epochs 3501
```



