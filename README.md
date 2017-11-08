##implement of DoReFaNet with tensorflow based on cifar10 dataset


the link of the paper is:https://arxiv.org/abs/1606.06160

My implementation is based on work in : https://github.com/AngusG/tensorflow-xnor-bnn & https://github.com/skoppula/dorefa-net.git


## Data
This implementation supports cifar10/cifar100  

## Dependencies
tensorflow version 1.2.1

## Training

* Train cifar10 model using gpu:
 Full presion:
 	python main_full.py     
 if you want to train your own DoReFanet
	python main_for_DoReFa.py

 Dorefanet:
	weight: 1 bit output:2 bits
	epoches: 100
	learning rate:0.001
	accuracy:85.3%


	weight: 2 bits output:2 bits
	epoches: 100
	learning rate:0.001
	accuracy:85.5%

	weight: 1 bit output:3 bits
	epoches: 100
	learning rate:0.001
	accuracy:85.5%

BNN:

	weight: 1 bit(-1,1) output:1 bits(-1,1)
	epoches: 100
	learning rate:0.001
	accuracy:84.1%



* Train cifar10 model using cpu:
 if you did not own a GPU which can speed up the training, you just need to change the GPU in main.py into True

## Results
Cifar10 should reach at least 88% top-1 accuracy






