# Trust Your Model:
## Iterative Label Improvement and Robust Training by Confidence Based Filtering and Dataset Partitioning

```This code is accompanying the publication named above and is intended to enable the reproduction of the experiments done for and described in this publication.```

## Installation instructions:

1) Setup folder structure<br/>
``` ./setup_dirs ```
```
Note: If you want to use tinyimagenet you need to download the files first and point src/dataset/tinyimagenet.py to the respective folder.
```

.<br/>
├── models - weights of the trained models<br/>
├── reports - accuracy files<br/>
│├── figures<br/>
│└── history<br/>
└── ili<br/>
&nbsp;&nbsp;├── datasets - custom datasets !adjust directory to actual files !<br/>
&nbsp;&nbsp;├── models - config, helpers, etc.<br/>
&nbsp;&nbsp;└── test - some optional tests using pytest<br/>


2) Install dependencies
* python3 (tested with version 3.5.2)
* If using virtual environment:
  * ```python -m venv ili_env```
  * ```source ili_env/bin/activate```
* ``` pip install -r requirements.txt ```

3) Run trainings
<br/>
<br/>

```
Note: the current version does *not* allow to combine: mnist-data with ResNet
```

### To get a baseline trained on noisy data

``` python train.py dataset_name error_type model_name noise_fraction run_index --AUG --SAVE --SAVEHIST```
* dataset_name - one of: "mnist" | "cifar10" | "cifar100" | "tinyimagenet"
* error_type - one of: "bias" | "random"
* model_name - one of: "mnist_cnn" | "cifar_cnn" | "resnet32" | "resnet50"
<br/>

* noise_fraction - noise fraction to use: float
* run_idx - index of run, int
<br/>

* --AUG - activate data augmentation
* --SAVE - save model weights after training
* --SAVEHIST - save history after training

E.g.<br/>
``` python train.py mnist random mnist_cnn 0.5 0 --AUG --SAVE --SAVEHIST```
<br/>
<br/>
### To do the ILI experiments
``` python train_ILI.py dataset_name error_type model_name mode (th) num_iter noise_fraction --AUG --SAVE --SAVEHIST```
* dataset_name - one of: "mnist" | "cifar10" | "cifar100" | "tinyimagenet"
* error_type - one of: "bias" | "random"
* model_name - one of: "mnist_cnn" | "cifar_cnn" | "resnet32" | "resnet50"
<br/>

* mode - one of: "confidence threshold (float < 1.0)" | "plain", e.g. >> confidence 0.3
<br/>

* num_iter - number of ILI iterations: int, default: 10
* noise_fraction - noise fraction to use: float
<br/>

* --AUG - activate data augmentation
* --SAVE - save model weights after training
* --SAVEHIST - save history after training

E.g.<br/>
``` python train_ILI.py mnist random mnist_cnn confidence 0.3 10 0.5 --AUG --SAVE --SAVEHIST```
<br/>
<br/>
### To do the opILI experiments
``` python train_opILI.py dataset error_type model_name mode num_iter noise_fraction --SAVE --SAVEHIST```
* dataset_name - one of: "mnist" | "cifar10" | "cifar100" | "tinyimagenet"
* error_type - one of: "bias" | "random"
* model_name - one of: "mnist_cnn" | "cifar_cnn" | "resnet32" | "resnet50"
<br/>

* mode - one of: "confidence threshold (float < 1.0)" | "plain", e.g. >> confidence 0.3
<br/>

* num_iter - number of ILI iterations: int, default: 10
* noise_fraction - noise fraction to use: float
<br/>

* --AUG - activate data augmentation
* --SAVE - save model weights after training
* --SAVEHIST - save history after training

E.g.<br/>
``` python train_opILI.py mnist random mnist_cnn confidence 0.3 10 0.5 --AUG --SAVE --SAVEHIST```

# Citation
If you use our approach in your research, we would be happy if you cite us:
```
@article{haase2020iterative,
  title={Iterative Label Improvement and Robust Training by Confidence Based Filtering and Dataset Partitioning},
  author={Haase-Sch{\"u}tz, Christian and Stal, Rainer and Hertlein, Heinz and Sick, Bernhard},
  journal={arXiv preprint arXiv:2002.02705},
  year={2020}
}
```

# License
```
(c) Robert Bosch GmbH 2019-2021. All rights reserved.
This software is solely developed for and published as part of the publication [arxiv](https://arxiv.org/pdf/2002.02705.pdf).
You are only allowed to use this code to reproduce the experiments presented in the named publication for scientific purposes.
```
