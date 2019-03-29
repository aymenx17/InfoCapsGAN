# FullyCapsGAN

### Introduction

In this work, I implemented a generative adversarial network based on the theories of capsules, dynamic routing by agreement and mutual information. The GAN is for the most part symmetric, given the similarity in the architecture between the generator and discriminator networks. This project pursues the idea of a path toward unsupervised classification of MNIST digits, achieving an accuracy much close to a comparable baseline classifier trained in a supervised manner. The two approaches would share the same encoder architecture. However in the unsupervised approach, at least during training stage, the architecture of the trainable infrastructure required is more complex,  harder to deal with, due to training stability and an accentuated stochastic behavior.

During training, in order for the discriminator to act as a classifier while capturing the class information from the unlabeled dataset, a random set of class labels, is given as input to the generator, whereas that exact set is used afterwards to penalize the discriminator. Hence the mutual information technique.



### Network Architecture

![Picture of network architecture](https://github.com/aymenx17/FullyCapsGAN/blob/master/project_images/FullyCapsGAN.png)


### Generated Images

##### Training on unlabeled dataset at Epoch 17:

![epoch: 17](https://github.com/aymenx17/FullyCapsGAN/blob/master/project_images/generated-17-500.png)
![epoch: 17](https://github.com/aymenx17/FullyCapsGAN/blob/master/project_images/generated-17-600.png)


### Environments

- Pytorch 1.0
- Python 3.6
- opencv 3.4.1
- numpy  1.16.2

For a more comprehensive list of required packages you may refer to the file conda_environment.yaml
You may also use conda package manager to recreate that specific working environment, with the following command:
```python
conda env create -f conda_environment.yaml
```

### Training

```python
python main.py
```
