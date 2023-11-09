# InfoCapsGAN

### Introduction

In this work, I implemented a generative adversarial network based on the theories of capsules, dynamic routing by agreement and mutual information. The GAN is for the most part symmetric, given the similarity in the architecture between the generator and discriminator networks. This project pursues the idea of a path toward unsupervised classification of MNIST digits, achieving an accuracy much close and comparable to baseline classifier trained in a supervised manner; where the two approaches would share the same encoder architecture. However in the unsupervised approach, at least during training stage, the architecture of the trainable infrastructure required is more complex,  harder to deal with, due to training stability and an accentuated stochastic behavior.

During training, in order for the discriminator to act as a classifier while capturing the class information from the unlabeled dataset, a random set of class labels, is given as input to the generator, whereas that exact set is used afterwards to penalize the discriminator. Hence the mutual information technique.

[Link to a pdf report](https://drive.google.com/open?id=16-0xJ0aDaJ_7UeneP4apcntHvhvQOGyX)


### Network Architecture

![Picture of network architecture](https://github.com/aymenx17/InfoCapsGAN/blob/master/project_images/InfoCapsGAN.png)


### Environment

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



### Branch: mimic

The current code implementation on the branch master reflects the architecture shown above in this repository. In this branch, however, the network design is not aligned with the final objective of this project, whereas it is in the branch called mimic.

Unlike in the standard architecture as in GAN theory where the input to the generator is randomly sampled, in here the input is the directly correlated to the output of the discriminator. More specifically in this implementation the last capsule layer of discrimination is provided as input to the generator, and currently the class info is not integrated in the network.

#### Generated Images

###### Training on unlabeled dataset at Epoch 17:

![epoch: 17](https://github.com/aymenx17/InfoCapsGAN/blob/master/project_images/generated-18-500.png)
![epoch: 17](https://github.com/aymenx17/InfoCapsGAN/blob/master/project_images/generated-18-600.png)


### Branch: master

The input to the generator is the concatenation of a one dimensional tensor of continuous code (random normal distribution) of size 32,  and a second tensor of discrete code (random hot-encoded class info) of the same size of the number of classes, 10 in the case of MNIST.  


#### Generated Images

###### Training on unlabeled dataset at Epoch 17:

![epoch: 17](https://github.com/aymenx17/InfoCapsGAN/blob/master/project_images/generated-17-500.png)
![epoch: 17](https://github.com/aymenx17/InfoCapsGAN/blob/master/project_images/generated-17-600.png)





