# ESN_ninaPro_EMG

to run this code type the following in your Jupyter or Ipython console:

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
