# ESN_ninaPro_EMG

This project was done at the University of sheffield in collaboration with University of Zurich and ActiveAI funding project **[May 2022 - August 2022]**. This codebase presents preliminary results on classifying forearm and fingers' Electromyography (EMG) signals taken in intact subjects into distinct 12 finger movements; towards implementing a light-weight neural network model 
on a neuromorphic system to control a bionic hand. 

 ## Dataset: 
 We used the public dataset NinaPro [https://ninapro.hevs.ch/index.html] which comprises of EMG signals taken from 10 electrodes positioned on the subject's arm, as illustrated in their dataset documentation, as they are tasked to perform distinct finger and grabbing motions. 
Given the current stage of this project and the time constraint, we trained and tested our recurrent neural network model on a subset of the full dataset, Ninapro DB1. In particular, we used the EMG recordings from a single subject (intact) who was tasked with 12 finger movements (exercise A in NinaPro Documentation). 
 
 
 ## Recurrent neural network model: 
We implemented an Echo State Network model **[Jaeger (2001);Jaeger and Hass (2004)]** (ESN) to train on classifying input EMG time signals into 12 movements classes. ESN networks is a type of reservoir computing where the reservoir recurrent connections are initialized randomly and stay fixed throughout training, learning happens only in the output layer weights which connect the recurrently connected **N** reservoir neurons to the output class neurons (12 in our model), **Nout**.

Our model piepline composes of 2 main classes:

### A- NinaPro class: this class handles the loading of the data and preparing it into a standard Pytorch tensor form to input to the ESN network. It consists of 6 main steps:

1- Loading the subject and exercise EMG sequence data from the dataset directory on path.
   
2- For each finger move (including rest/no motion), concatenate consecutive time steps of the same finger move (target lable) into a single EMG instance: **L time sequence x K (10 electrodes)**.
   
3- Find the maximum sequence length in all EMG finger moves and pad the shorter sequences with EMG samples taken from the next rest/no finger movement sequence.
   
4- Create a mask to keep track of true EMG data and the padded parts, to ignore the padding sequence when training the ESN model.
   
5- Transform the EMG data arrays of the same length after padding into pytorch tensor: **N batch x K features x L time sequence**.
   
6- Split the torch tensor EMG and target labels data into training and testing sets: 70%-30%.  

  
###  B- Echo State Network (ESN) class: this class handles initializing the ESN network, training and testing the model on the input data.

1- ESN response: this method ensure initializing the reservoir with the echo state property and calculate the dynamical responses of the reservoir neurons. 
    
2- LR from response: this method learns the correct output weights given the ESN responses output. It does so by computing the loss between predicted and the correct class type of the input EMG time signal sample, and back propagates this error to change the network output weights and biases **Ws,bs**.   

 ## Run the code: 
 
A- Clone and unzip the repoistory.

B- Run Jupyter notebook from the repository directory path on your machine: '\LOCAL_REPOSITORY_DIRECTORY\ESN_ninaPro_EMG-main\ESN_ninaPro_EMG-main\code.

C-Run the following cells in Jupyter notebook

```python
## this code will import the data from EMG signals E1
## input: the goal is to classify each time sample from 10 electrodes/dimensions to a motion
## there are 10 repetitions per motion, use 70 for training and 30 for testing
## output: accuracy of classifying the testing test into these 12 output classes. 

import torch
import numpy as np
from torch import nn
from torch import optim
import scipy.io
import random
import math
import matplotlib.pyplot as plt
import os
import ESN_NinaPro_train_test
```


```python
## change 'C:/Users/abdelrahmann/Downloads/' to the path of the ESN_ninaPro_EMG-main repository on your machine
data_dir='C:/Users/abdelrahmann/Downloads/ESN_ninaPro_EMG-main/ESN_ninaPro_EMG-main/Data/'
```


```python
import zipfile
with zipfile.ZipFile(data_dir +'s1.zip', 'r') as zip_ref:
              zip_ref.extractall(data_dir+'unzipped_s1')
```


```python
current_directory = os.getcwd()
print(current_directory)
```


```python
db=ESN_NinaPro_train_test.NinaPro(data_dir+'unzipped_s1')

S_tr,S_te,Y_tr,Y_te,masktr,maskte=db.load()
```


```python
N=200      # no. of reservoir neurons
N_in=10    #no. of input signal dimensions (10 electrodes)
N_av=50    # parameter for determining the sparsity of the recurrent connections between the reservoir neurons
alpha=0.99 # weighing of the current input signal and reservoir excitations relative to the previous reservoir neurons' states in updating the current
           # reservoir neurons' states. 
rho=0.95 # spectral radius scale of the reservoir recurrent matrix
gamma=1 # scaling parameters for the input matrix weights
N_out=12 # no. of output neurons (12 finger movements)

## initialize the ESN model with the hyperparameters chosen above
esn=ESN_NinaPro_train_test.ESN(N,N_in,N_out,N_av,alpha,rho,gamma)

## compute the reservoir excitations given the input training EMG signals. 
X=esn.ESN_response(S_tr)
```


```python
## train the model for 500k trials and print the training and testing losses every 50k trials.
## print the final model accuracies.
Nbatches=500000  # total number of training trials
Ncheck=50000 # print losses values every Ncheck trials
eta=0.00001 # model learning rate
test_acc,training_acc,training_losses=esn.LR_from_response(S_tr,Y_tr,S_te,Y_te,Nbatches,eta,Ncheck,masktr,maskte)
```


```python
## plot subsampled training losses curve, every 10k trials ( you could pick other numbers)
import matplotlib.pyplot as plt
subsamples=10000
n_trials=Nbatches/subsamples
sliced_array=training_losses[::subsamples].tolist()
plt.plot(range(0,n_trials),sliced_array)
plt.xlabel("Training Batch")
plt.ylabel("Loss")
plt.title("Training Loss Over Batches")
plt.grid(True)
plt.show()
```



