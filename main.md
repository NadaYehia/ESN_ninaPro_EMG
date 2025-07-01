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
## change 'C:/Users/abdelrahmann/Downloads/' to the path of the repository on your machine
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
db=NinaPro(data_dir+'unzipped_s1')

S_tr,S_te,Y_tr,Y_te,masktr,maskte=db.load()
```


```python
N=200
N_in=10 
N_av=50 
alpha=0.99 
rho=1.5 
gamma=1 
N_out=12 
esn=ESN(N,N_in,N_out,N_av,alpha,rho,gamma)

X=esn.ESN_response(S_tr)
```


```python
test_acc,training_acc,training_losses=esn.LR_from_response(S_tr,Y_tr,S_te,Y_te,500000,0.00001,50000,masktr,maskte)
```


```python
print(len(training_losses))  # Check total size
#print((training_losses[::10000]))  # Check sampled size
print(np.any(np.isnan(training_losses)))  # Check for NaN values
print(np.any(np.isinf(training_losses)))  # Check for infinite values
```


```python
import matplotlib.pyplot as plt

sliced_array=training_losses[::10000].tolist()
plt.plot(range(0,50),sliced_array)
plt.xlabel("Training Batch")
plt.ylabel("Loss")
plt.title("Training Loss Over Batches")
plt.grid(True)
plt.show()
```
