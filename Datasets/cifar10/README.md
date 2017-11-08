##Tenary weight network implement using tensorflow
author: Yadongwei (XJTU)
Training Deep Neural Networks with Weights and Activations Constrained to +1,0  or -1.  implementation in tensorflow (https://arxiv.org/abs/1605.04711)

This is incomplete training example for BinaryNets using Binary-Backpropagation algorithm as explained in 
"Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained (1,-1) or (1,0,-1), 
on following datasets: Cifar10/100.


My implementation is based on work in : https://github.com/AngusG/tensorflow-xnor-bnn


## Data
This implementation supports cifar10/cifar100  

## Dependencies
tensorflow version 1.2.1

## Training

* Train cifar10 model using gpu:
 Full presion:
 	python main_full.py     
 accuracy: 
 	83.3%(10 epoches, learning rate:0.01) 
 	87.0%(50 epoches, learning rate:0.005)
 Binaried the weight and output
 	python main_for_bnn.py  
 accuracy: 
 	79.5%(10 epoches, learning rate:0.01) 
 	80% (50 epoches, learning rate:0.005)
 Binaried the weight:
 	python main_for_bnn1.py 
 accuracy: 
 	82.5% (10 epoches, learning rate:0.01)
 	86.1% (50 epoches, learning rate:0.005)
 Ternaried the weight:
 	python main.py          
 accuracy: 
 	83% (10 epoches, learning rate:0.01)
 	85.7% (50 epoches, learning rate:0.005)
* Train cifar10 model using cpu:
 if you did not own a GPU which can speed up the training, you just need to change the GPU in main.py into True

## Results
Cifar10 should reach at least 88% top-1 accuracy






