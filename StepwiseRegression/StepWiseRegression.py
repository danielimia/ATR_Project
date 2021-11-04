# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:59:56 2021

@author: aantonakis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def MultiDLineFit(SampleX,SampleY):
    #SampleX=Sample.drop(ColY,axis=1).values
    #SampleY=Sample[ColY]
    #1D polyfit function
    Ncoeff=SampleX.shape[1]
    #set up least squares equation
    A=np.zeros((Ncoeff,Ncoeff))
    B=np.zeros(Ncoeff)
    C=B*0
    for i in range(Ncoeff):
        for j in range(Ncoeff):
            A[i,j]=np.sum(SampleX[:,j]*SampleX[:,i])
        B[i]=np.sum(SampleY*SampleX[:,i])
    C=np.linalg.solve(A,B)
    C2=np.zeros((len(C),1))
    C2[:,0]=C*1
    Yh=np.dot(SampleX,C2)
    return C2,Yh

def GetPerformance(Yref,Yh,Nparams): #adjusted R2
    SSres=np.dot(Yref-Yh,Yref-Yh)
    SStot=np.dot(Yref-np.mean(Yref),Yref-np.mean(Yref))
    R2=1-SSres/SStot
    """
    N=len(Yref)
    adjR2=1-(1-R2)*(N-1)/(N-Nparams-1)
    """
    return R2#-0.0001*Nparams

def ForwardSelectionStep(df,Selected,ColumnsIN,ColumnOUT,Performance):
    #create reference Output
    Yref=df[ColumnOUT].to_numpy()
    #run column selection
    CoL_Performance=np.ones(len(ColumnsIN))*Performance-10 #initialize with worse value
    for i in range(len(ColumnsIN)):
        Column=ColumnsIN[i]
        if not (Column in Selected): #if column not already selected
            RegressionColumns=Selected.copy()+[Column] #add column to regression columns
            
            df_new_IN=df[RegressionColumns] #slice Dataframe
            df_new_IN['Bias']=np.ones(len(df_new_IN)) #add bias
            SampleX=df_new_IN.to_numpy() #convert inputs to numpy
            C,Yh=MultiDLineFit(SampleX,Yref) #calculate regression
            CoL_Performance[i]=GetPerformance(Yref,Yh[:,0],len(RegressionColumns)+1) #get performance
            """
            plt.figure()
            plt.plot(Yref,Yh,'k.')
            plt.grid('on')
            plt.axis('equal')
            plt.title(str(RegressionColumns))
            """
    if np.max(CoL_Performance)>Performance: #if better performance has been achieved
        Selected.append(ColumnsIN[np.argmax(CoL_Performance)])
        Performance=np.max(CoL_Performance)
    
    return Selected,Performance,CoL_Performance

def BackwardRejectionStep(df,Selected,ColumnsIN,ColumnOUT,Performance):
    #create reference Output
    Yref=df[ColumnOUT].to_numpy()
    #run column selection
    CoL_Performance=np.ones(len(Selected))*Performance-10 #initialize with worse value
    for i in range(len(Selected)):
        Column=Selected[i]
        
        RegressionColumns=Selected.copy()
        RegressionColumns.remove(Column)#add column to regression columns
        
        df_new_IN=df[RegressionColumns] #slice Dataframe
        df_new_IN['Bias']=np.ones(len(df_new_IN)) #add bias
        SampleX=df_new_IN.to_numpy() #convert inputs to numpy
        C,Yh=MultiDLineFit(SampleX,Yref) #calculate regression
        CoL_Performance[i]=GetPerformance(Yref,Yh[:,0],len(RegressionColumns)+1) #get performance
        """
        plt.figure()
        plt.plot(Yref,Yh,'k.')
        plt.grid('on')
        plt.axis('equal')
        plt.title(str(RegressionColumns))
        """
    if np.max(CoL_Performance)>Performance: #if better performance has been achieved
        Selected.remove(ColumnsIN[np.argmax(CoL_Performance)])
        Performance=np.max(CoL_Performance)
    
    return Selected,Performance,CoL_Performance

#----Test Run----------------------
plt.close('all')

#read data
df=pd.read_csv('SampleAOA2.csv',sep=';')

#initialize stepwise selection process
Selected=[] #selected regression parameters [empty]
ColumnsIN=df.columns.to_list() #df columns corresponding to inputs
ColumnsIN.remove('[X0_0]')
ColumnOUT='[X0_0]' #output column
Performance=0.0 #R2 performance
Performance0=-100.0 
RegrList,k=[],0 #Regression evolution tracking
while Performance-Performance0>0 and k<10: #while performance is improving
    print(k,Selected,Performance)
    RegrList=RegrList+[[Selected.copy(),Performance*1]] #store current regression
    Performance0=Performance*1
    #forward selection step
    Selected,Performance,CoL_Performance=ForwardSelectionStep(df,Selected,ColumnsIN,ColumnOUT,Performance)
    #backward rejection step
    Selected,Performance,CoL_Performance=BackwardRejectionStep(df,Selected,ColumnsIN,ColumnOUT,Performance)
    k+=1
    
#plot results
plt.figure()
plt.plot([np.log10(1-x[1]) for x in RegrList],'-k.')
for i in range(len(RegrList)):
    plt.text(i,np.log10(1-RegrList[i][1]),str(RegrList[i][0]),ha='center',va='bottom')
plt.xlabel('No of Regression Variables')
plt.ylabel('log10(R2)')
plt.grid('on')
