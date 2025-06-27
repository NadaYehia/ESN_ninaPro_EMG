# ESN_ninaPro_EMG

This project was done at the University of sheffield in collaboration with University of Zurich and ActiveAI funding project [May 2022 - August 2022]. This codebase presents preliminary results
on classifying forearm and fingers' Electromyography (EMG) signals taken in intact subjects into distinct 12 finger movements; towards implementing a light-weight neural network model 
on a neuromorphic system to control a bionic hand. 

 ## Dataset: 
 We used the public dataset NinaPro [https://ninapro.hevs.ch/index.html] which comprises of EMG signals taken from 10 electrodes positioned on the subject's arm, as illustrated in their dataset documentation, as they are tasked to perform distinct finger and grabbing motions. 
Given the current stage of this project and the time constraint, we trained and tested our recurrent neural network model on a subset of the full dataset, Ninapro DB1. In particular, we used the EMG recordings from a single subject (intact) who was tasked with 12 finger movements (exercise A in NinaPro Documentation). 
 
 
 ## Recurrent neural network model: 


- NinaPro class:
- Echo State Network (ESN) class:
    - ESN response:
    - LR from response: 




 ## Run the code: 
  Run this code type the following in your Jupyter or Ipython console:

- runfile('/YOUR WORKING DIRECTORY/ESN_NinaPro_train_test.py', wdir=[YOUR WORKING DIRECTORY])

- db=NinaPro('/DATABASE_dIRECTORY/DB_NinaPro_1')

- S_tr,S_te,Y_tr,Y_te,masktr,maskte=db.load()
- N=200
  N_in=10
  N_av=10
  alpha=0.9
  rho=0.99
  gamma=0.1
  N_out=12
  esn=ESN(N,N_in,N_out,N_av,alpha,rho,gamma)
  
  - X=esn.ESN_response(S_tr)
  
  Here I am using learning rate, eta=0.0001
                  number of epochs= 500000
                  [feel free to tune these hyperparameters]
                  *Further exploration of them is needed*
  
  - esn.LR_from_response(S_tr,Y_tr,S_te,Y_te,500000,0.0001,50000,masktr,maskte)
