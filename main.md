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

    C:\Users\abdelrahmann\Downloads\ESN_ninaPro_EMG-main\ESN_ninaPro_EMG-main\code
    


```python
db=NinaPro(data_dir+'unzipped_s1')

S_tr,S_te,Y_tr,Y_te,masktr,maskte=db.load()
```

    1176
    3
    


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

    iteration 0 training loss: [2.48745275]
    Iteration 0 loss:  tensor(7.0713, grad_fn=<NegBackward0>)
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-73-eabf7d088bbe> in <module>
    ----> 1 test_acc,training_acc,training_losses=esn.LR_from_response(S_tr,Y_tr,S_te,Y_te,500000,0.00001,50000,masktr,maskte)
    

    <ipython-input-61-49264790b8cc> in LR_from_response(self, S_tr, Y_tr, S_te, Y_te, N_batch, eta, N_check, masktr, maskte)
         96             loss=nn.CrossEntropyLoss()
         97             outputL=loss(pred,target)'''
    ---> 98             outputL.backward()
         99             self.opt.step()
        100             self.opt.zero_grad()
    

    ~\AppData\Local\anaconda3\envs\tensorflow_env\lib\site-packages\torch\_tensor.py in backward(self, gradient, retain_graph, create_graph, inputs)
        305                 create_graph=create_graph,
        306                 inputs=inputs)
    --> 307         torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
        308 
        309     def register_hook(self, hook):
    

    ~\AppData\Local\anaconda3\envs\tensorflow_env\lib\site-packages\torch\autograd\__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
        154     Variable._execution_engine.run_backward(
        155         tensors, grad_tensors_, retain_graph, create_graph, inputs,
    --> 156         allow_unreachable=True, accumulate_grad=True)  # allow_unreachable flag
        157 
        158 
    

    KeyboardInterrupt: 



```python
print(len(training_losses))  # Check total size
#print((training_losses[::10000]))  # Check sampled size
print(np.any(np.isnan(training_losses)))  # Check for NaN values
print(np.any(np.isinf(training_losses)))  # Check for infinite values
```

    500000
    True
    False
    


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


    
![png](output_8_0.png)
    

