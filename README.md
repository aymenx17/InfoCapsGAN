# FullyCapsGan

###Introduction

In this work, I implemented a generative adversarial network based on the theories of capsules, dynamic routing by agreement and mutual information. The GAN is for the most part symmetric, given the similarity in the architecture between the generator and discriminator networks. This project pursues the idea of a way toward unsupervised classification of MNIST digits,  where in a post-training stage of the full network, GAN, the discriminator is then taken as encoder along with a classifier, and the objective is to enable the achievement of similar accuracy of an equivalent encoder trained in a supervised fashion. During training, in order for the discriminator to act as a classifier while capturing the class information from the unlabeled dataset, a random set of class info, used afterwards as ground truth, is given as input to the generator. Hence the mutual information technique. 


### Network Architecture

![Picture](https://github.com/aymenx17/FullyCapsGAN/blob/master/FullyCapsGan.png)




