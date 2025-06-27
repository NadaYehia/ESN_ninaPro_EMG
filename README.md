# ESN_ninaPro_EMG

This project was done at the University of sheffield in collaboration with University of Zurich and ActiveAI funding project [May 2022 - August 2022]. This codebase presents preliminary results
on classifying forearm and fingers' Electromyography (EMG) signals taken in intact subjects into distinct 12 finger movements; towards implementing a light-weight neural network model 
on a neuromorphic system to control a bionic hand. 

 ## Dataset: 
 We used the public dataset NinaPro [https://ninapro.hevs.ch/index.html] which comprises of EMG signals taken from 10 electrodes positioned on the subject's arm, as illustrated in their dataset documentation, as they are tasked to perform distinct finger and grabbing motions. 
Given the current stage of this project and the time constraint, we trained and tested our recurrent neural network model on a subset of the full dataset, Ninapro DB1. In particular, we used the EMG recordings from a single subject (intact) who was tasked with 12 finger movements (exercise A in NinaPro Documentation). 
 
 
 ## Recurrent neural network model: 
We implemented an Echo State Network model (ESN) to train on classifying input EMG time signals into 12 movements classes. ESN networks is a type of reservoir computing where the reservoir recurrent connections are initialized randomly and stay fixed throughout training, learning happens only in the output layer weights which connect the reservoir neurons **N** to the output class neurons (12 in our model) **Nout**.
our model piepline composes of 2 main classes:

### A- NinaPro class: this class handles the loading of the data and preparing it into a standard Pytorch tensor form to input to the ESN network. It consists of 6 main steps:

1- Loading the subject and exercise EMG sequence data from the dataset directory on path.
   
2- For each finger move (including rest/no motion), concatenate consecutive time steps of the same finger move (target lable) into a single EMG instance: **Nseq x F (10 electrodes)**.
   
3- Find the maximum sequence length in all EMG finger moves and pad the shorter sequences with EMG samples taken from the next rest/no finger movement sequence.
   
4- Create a mask to keep track of true EMG data and the padded parts, to ignore the padding sequence when training the ESN model.
   
5- Transform the EMG data arrays of the same length after padding into pytorch tensor: **N batch x L seq x K features**.
   
6- Split the torch tensor EMG and target labels data into training and testing sets: 70%-30%.  

  
###  B- Echo State Network (ESN) class: this class handles initializing the ESN network, training and testing the model on the input data.

1- ESN response: this method ensure initializing the reservoir with the echo state property and calculate the dynamical responses of the reservoir neurons. 
    
2- LR from response: this method learns the correct output weights given the ESN responses output. It does so by computing the loss between predicted and the correct class type of the input EMG time signal sample, and back propagates this error to change the network output weights and biases **Ws,bs**.   

 ## Run the code: 
To run this code type the following in your Jupyter or Ipython console:
Steps: 

1- runfile('/YOUR WORKING DIRECTORY/ESN_NinaPro_train_test.py', wdir=[YOUR WORKING DIRECTORY])

**Initialize the Ninapro class given the data path, data_dir is the path for the Data folder in this repo on your computer. 
  s1.zip is the dataset for one subject in 3 sets of different finger movements exercises (E1,E2,E3): 52 different movements in total** 
  
2a- python

    import zipfile
    
    with zipfile.ZipFile(data_dir +'s1.zip', 'r') as zip_ref:
    
              zip_ref.extractall(data_dir+'unzipped_s1')
              
2b- db=NinaPro(data_dir+'unzipped_s1')

 **create the training and testing datasets**
 
3- S_tr,S_te,Y_tr,Y_te,masktr,maskte=db.load()

**initialize the ESN model and set its hyperparameters like, the number of reservoir neurons (N), number of input neurons (Nin), number of output neurons (Nout), and reservoir recurrency parameters (alpha and gamma), and the scaling factor for the reservoir matrix spectral radius to ensure echo state property(rho)** 

4- N=200
  N_in=10
  N_av=10
  alpha=0.9
  rho=0.99
  gamma=0.1
  N_out=12
  esn=ESN(N,N_in,N_out,N_av,alpha,rho,gamma)

  **calculate the reservoir dynamic resposnes given the input training EMG sequences**
  
  5- X=esn.ESN_response(S_tr)

  **train the network model and set training hyperparameters like learning rate (eta) and number of epochs (N epochs)**
  
  Here I am using learning rate, eta=0.0001
                  number of epochs= 500000
                  [feel free to tune these hyperparameters]
                  *Further exploration of them is needed*
  
  6- esn.LR_from_response(S_tr,Y_tr,S_te,Y_te,500000,0.0001,50000,masktr,maskte)
