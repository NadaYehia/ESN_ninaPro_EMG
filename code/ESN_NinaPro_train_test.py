# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 00:35:55 2022

@author: lucam & nada
"""
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

# This class will contain the data from ninapro web.
# for Exercise 1 (as a start)
# target classes are : 0- 12 (4 fingers felxions and extenstions) + thumb 3 motions + 12 (no motion)
 
class NinaPro:
    def __init__(self,DataPath):
        self.DataPath=DataPath
        self.DataSubjects=[]
        self.EMG=dict()
        self.Target=dict()
        
        self.EMGTrain=dict()
        self.TargetTrain=dict()
        self.EMGTest=dict()
        self.TargetTest=dict()
        self.emgFullData=[[[]]]
        self.targetFullData=[]
    
        
    def load(self):   
        for j in range(1):
            SubjectjE1='/S'+str(j+1)+'_A1_E1.mat'
            self.DataSubjects = scipy.io.loadmat(self.DataPath+SubjectjE1)
            
            # load the restimulus array: target movements classes
            # load the emg matrix: time x 10 electrodes
            
            self.EMG[str(j)]=self.DataSubjects['emg']
            self.Target[str(j)]=self.DataSubjects['restimulus']
            
            [self.emgFullData,self.targetFullData]=self.concateunateSequences(self.EMG[str(j)],self.Target[str(j)])
            [self.emgFullDataTensor,self.maskPerSeq]=self.fromListToTensor(self.emgFullData, self.targetFullData)
            self.targetFullData=np.array(self.targetFullData[120:240])
            
            # array to torch tensor
            self.emgFullDataTensor=torch.from_numpy(self.emgFullDataTensor)
            self.targetFullData=torch.from_numpy(self.targetFullData)
            self.maskPerSeq=torch.from_numpy(self.maskPerSeq)
            
            [self.S_tr,self.S_te,self.Y_tr,self.Y_te,self.maskTr,self.maskTe]=self.split_train_test(self.emgFullDataTensor,self.targetFullData,self.maskPerSeq)
        
        
        return self.S_tr,self.S_te,self.Y_tr,self.Y_te,self.maskTr,self.maskTe
    
        # split data into training and testing, there are 10 repetitions per motion
    def concateunateSequences(self,emg,target):
        self.emgSub=emg
        self.targetSub=target
        
        self.emgFullDataset= []
        self.targetFullDataset=[]
        
        TrainingSize=math.floor(0.7*self.targetSub.shape[0])
        TestingSize= math.floor(0.3*self.targetSub.shape[0])
                
        self.emgSubTrain=[]
        self.targetSubTrain=[]
        
        self.emgSubTest=[]
        self.targetSubTest=[]
        
        
        # tensor contains the indices selected at random for the training samples: 
        # its size is 70% of the datasamples length.
        rep=0
       
        for myClass in range (13):
            
            # loop in the emg data, for everytime step if class=myClass
            # then store this sequence till the myclass isnt encountered anymore.
            # store this in rep1, dim 0 of the dataset.
            old_rep=rep
            t=0
            for tim in range (self.emgSub.shape[0]):
                
                
                if (myClass==self.targetSub[tim]):
                    if(t==0): 
                         
                        old_t=tim
                    #print(old_t)
                    t=t+1
                    
                elif(t!=0):
                    #end of a sequence
                    self.emgFullDataset.append(self.emgSub[old_t:old_t+t][:])
                    t=0
                    self.targetFullDataset.append(myClass)
                    rep=rep+1

        
        return self.emgFullDataset,self.targetFullDataset                 


    def findMaxSeqLen(self,emgFullData):
        self.maxL=0
        # each movement sequence is followed by a rest sequence, therefore there are 120 rest sequences for 120 finger movements sequences
        # this number can be calculated in the future; it does not need to be hardcoded.
        offset=120
        
        #loop over the gestures sequences:120-240, 12 movements in E1 (without the rest/no finger movements sequences) x 10 repetitions (each movement)
        for mySeq in range(120):
            #  find the biggest sequence length: sequence is made of actual finger movement sequence + a rest sequence
            myLen=len(emgFullData[offset+mySeq])
            padL=len(emgFullData[mySeq])
            
            if((myLen+padL)>self.maxL):
                self.maxL=(myLen+padL)
        
        return self.maxL
        

    def fromListToTensor(self,emgFullData,targetFullData):
        
        self.maxSeqL=self.findMaxSeqLen(emgFullData)
        print(self.maxSeqL)
        self.emgFullDataTensor=np.zeros([120,10,self.maxSeqL])
        self.offset=120
        self.maskPerSeq=np.ones([120,self.maxSeqL])
        
        for mySeq in range(120):
            
            # transpose each LxK sequence to K xL: no. features x Time steps
            self.sequence=emgFullData[self.offset+mySeq]
            self.sequence=np.transpose(np.array(self.sequence),[1,0])

            # transpose each LxK sequence to K xL: no. features x Time steps
            self.paddingRest=emgFullData[mySeq]
            self.paddingRest=np.transpose(np.array(self.paddingRest),[1,0])
            
            #concatenate each finger movement sequence at the end with the rest padding
            self.FullSeq=np.concatenate((self.sequence,self.paddingRest),axis=1)
            self.nonEmptyL=self.FullSeq.shape[1]
            
            self.emgFullDataTensor[mySeq,:,0:self.nonEmptyL]=self.FullSeq

            # a mask of 1s at the indices of the actual EMG signal and 0s at the 0 padding locations.
            self.maskPerSeq[mySeq,self.nonEmptyL:self.maxSeqL]=np.multiply(self.maskPerSeq[mySeq,self.nonEmptyL:self.maxSeqL],0)
            
        return self.emgFullDataTensor,self.maskPerSeq

    def split_train_test(self, emgFullDataTensor,target,mask): 
        # now we have the data in a standard pytorch tensor of size:
        # N sequences (120 - 12 finger movements x 10 repetitions each) x K (10 electrodes) x T (sequence time length)
        
        T= emgFullDataTensor.size(dim=2)
        self.reps=10
        
        # split the data into 70% training, 30% testing. 
        # allocate empty tensors for the training signals, test signals, their masks (padding..not padding)
        # and their corresponding target finger movement type (1-12)
        self.S_tr= torch.empty((round(120*0.7),10,T))
        self.S_te=torch.empty((round(120*0.3),10,T))
        self.maskTr=torch.empty((round(120*0.7),T))
        self.maskTe=torch.empty((round(120*0.3),T))
        self.Y_tr=torch.empty(round(120*0.7))
        self.Y_te=torch.empty(round(120*0.3))
        
        self.trainingSeqperClass=int(0.7*self.reps)
        self.testingSeqperClass=int(0.3*self.reps)
        print(self.testingSeqperClass)
        seq=0
        s_tr=0
        s_te=0
        for myClass in range(1,13):
            #print((0.7*self.reps))
            trainingInd= random.sample(range(seq,seq+self.reps), int(0.7*self.reps))
            
            testingInd= list(set(range(seq,seq+self.reps))-set(trainingInd))
            
            self.S_tr[s_tr:s_tr+(self.trainingSeqperClass),:,:]=emgFullDataTensor[trainingInd,:,:]
            self.maskTr[s_tr:s_tr+(self.trainingSeqperClass),:]=mask[trainingInd,:]
            self.Y_tr[s_tr:s_tr+(self.trainingSeqperClass)]=target[trainingInd]
           
            
            self.S_te[s_te:s_te+(self.testingSeqperClass),:,:]=emgFullDataTensor[testingInd,:,:]
            self.maskTe[s_te:s_te+(self.testingSeqperClass),:]=mask[testingInd,:]
            self.Y_te[s_te:s_te+(self.testingSeqperClass)]=target[testingInd]
            
            seq=seq+self.reps
            s_tr=s_tr+(self.trainingSeqperClass)
            s_te=s_te+(self.testingSeqperClass)
    
    
    
        self.Y_tr= self.Y_tr.unsqueeze(1)
        self.Y_te= self.Y_te.unsqueeze(1)
        self.Y_tr=self.Y_tr.expand(round(120*0.7),T)
        self.Y_te=self.Y_te.expand(round(120*0.3),T)
        
        #self.Y_tr=self.Y_tr.long()-1
        #self.Y_te=self.Y_te.long()-1

                           
        return self.S_tr,self.S_te,self.Y_tr,self.Y_te,self.maskTr,self.maskTe
    
    

class ESN(nn.Module):
    
    def __init__(self,N,N_in,N_out,N_av,alpha,rho,gamma):
        super().__init__()
        
        
        self.N=N
        self.alpha=alpha
        self.rho=rho
        self.N_av=N_av
        self.N_in=N_in
        self.gamma=gamma
        self.N_out=N_out
        
        diluition=1-N_av/N
        W=np.random.uniform(-1,1,[N,N])
        W=W*(np.random.uniform(0,1,[N,N])>diluition)
        eig=np.linalg.eigvals(W)
        self.W=torch.from_numpy(self.rho*W/(np.max(np.absolute(eig)))).float()
        
        
        self.x=[]
        
        if self.N_in==1:
            
            self.W_in=2*np.random.randint(0,2,[self.N_in,self.N])-1
            self.W_in=torch.from_numpy(self.W_in*self.gamma).float()
            
            
        else:
            
            self.W_in=np.random.randn(self.N_in,self.N)
            self.W_in=torch.from_numpy(self.gamma*self.W_in).float()
            
        
        self.Ws=[]
        self.bs=[]
        self.theta=[]
        
        self.Ws.append(nn.Parameter( (2*torch.rand([self.N,self.N_out])-1)/(self.N)) )
        self.bs.append(nn.Parameter(torch.zeros([self.N_out])))

        
        self.opt=[]
        
        
    def Reset(self,s):
        
        batch_size=np.shape(s)[0]
        self.x=torch.zeros([batch_size,self.N])
        
    def ESN_1step(self,s):
        
        self.x=(1-self.alpha)*self.x+self.alpha*torch.tanh(torch.matmul(s,self.W_in)+torch.matmul(self.x,self.W))
        
    def ESN_response(self,Input):
        
        T=Input.shape[2]
        X=torch.zeros(Input.shape[0],self.N,T)
        
        self.Reset(Input[:,0])
        
        for t in range(T):
            
            self.ESN_1step(Input[:,:,t])
            X[:,:,t]=torch.clone(self.x)
            
        return X
    
    def LR_from_response(self,S_tr,Y_tr,S_te,Y_te,N_batch,eta,N_check,masktr,maskte):
        
        batch_size=S_tr.shape[0]
        N_tr=S_tr.shape[2]
        
        X_tr=self.ESN_response(S_tr)
        X_te=self.ESN_response(S_te)
        
        self.opt=optim.Adam(self.Ws+self.bs, lr=eta)
        training_losses=np.empty((N_batch,1))
     
        # number of training times
        for n in range(N_batch):
            
            ind_t=np.random.randint(low=0.,high=N_tr)
            
            #seqx12
            y=torch.matmul(X_tr[:,:,ind_t],self.Ws[0])+self.bs[0]
            
            #seq
            target=Y_tr[:,ind_t]
            target=target[ (masktr[:,ind_t]>0)]
            
            target=(target-1).long()
            
            y=y[masktr[:,ind_t]>0,:]
           
            outputL=self.myLoss(y,target)
            outputL.backward()
            
            self.opt.step()
            self.opt.zero_grad()
            
            training_losses[n]=outputL.item()
            if n%N_check==0:
                
                myLoss=self.Test(X_te,Y_te,maskte)
                print('iteration',n,'training loss:',training_losses[n])
                print('Iteration', n, 'loss: ', myLoss)
                
        testingAcc= self.Accuracy(X_te, Y_te, maskte)
        trainingAcc= self.Accuracy(X_tr, Y_tr, masktr)
        
        print('testing Accuracy:', testingAcc)#
        print('training Accuracy:', trainingAcc)
        return testingAcc,trainingAcc,training_losses       
            
    def Test(self,X_te,Y_te,maskte):
        
         Y=torch.transpose(torch.matmul(torch.transpose(X_te[:,:,:],1,2),self.Ws[0])+self.bs[0],1,2)
         
         myLoss=self.myLossTe(Y,Y_te,maskte)
         return myLoss
    
    def Accuracy(self,X_,Y_,mask):
        
         Y=torch.transpose(torch.matmul(torch.transpose(X_[:,:,:],1,2),self.Ws[0])+self.bs[0],1,2)
         
         #class number of every sequence and time sample
         Y=torch.argmax(Y,dim=1)
         
         Y_=torch.flatten(Y_-1)
         #print(Y_.shape)
         #print(Y.shape)
         
         correctClass=(torch.flatten(Y)==Y_)
         mask=torch.flatten(mask)
         
         CorrectClass_1=torch.mul(correctClass,mask)
         
         noCorrectClass=(CorrectClass_1[CorrectClass_1==True]).shape[0]
         #masked number of samples
         N_x_T=(mask[mask>0]).shape[0]
         
         accuracy= noCorrectClass/(N_x_T)
         
         return accuracy
            
    def softmax(self,x): return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)
    
    def nl(self,input, target): 
        lo=-input[range(target.shape[0]), target].log().mean()
        return lo
    
    def nlTe(self,input, target,maskte): 
        
        target=(target-1).long()
        # this is a tesnor of 0-35
        seqDim=torch.tensor(range(target.shape[0]))
        # expand this dim for the T samples: make it also 35 in dim0 x T in dim1
        seqDim=seqDim.expand(target.shape[1],seqDim.shape[0])
        #print(seqDim.shape)
        #flatten it out to a 1D tensor again
        seqDim=torch.flatten(seqDim.transpose(1,0))
        
        # seqdim is 1d tensor that have values of each sequence repeated for T times
        # for the time dimension, repeat it for the size of the samples.
        timDim=torch.tensor(range(target.shape[1]))
        timDim=torch.flatten(timDim.repeat(1,target.shape[0]))
        lo=input[seqDim, torch.flatten(target),timDim]
        
        lo=torch.mul(lo,torch.flatten(maskte))
        lo=lo[lo>0]
        lo=-lo.log().mean()
        return lo    
    
        
    def myLoss(self,y,target):
        
        pred=self.softmax(y)
        loss= self.nl(pred,target)
        return loss
    
    def myLossTe(self,y,target,maskte):
        
        pred=self.softmax(y)
        loss= self.nlTe(pred,target,maskte)
        return loss    
         
     
