# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 13:24:29 2021

@author: aantonakis
"""

import numpy as np
import pandas as pd


def Combinations(Nvars,P): #including duplicate (self-multiplication) entries
    Combin=[]
    Markers=np.arange(P)*0
    Combin.append(Markers*1)
    while Markers[0]<Nvars-1:
        Markers[-1]+=1
        for j in range(P-1,0,-1):
            if Markers[j]>Nvars-1:
                Markers[j-1]+=1
                for k in range(j,P):
                    Markers[k]=Markers[k-1]
        Combin.append(Markers*1)
    return Combin

def AddProductTerms(df,P):
    Columns=df.columns.to_list()
    Nvars=len(Columns)
    Combin=Combinations(Nvars,P)
    #add self
    for combination in Combin:
        Data=np.ones(len(df))
        
        Name=''
        for item in combination:
            Name=Name+Columns[item]
            Data=Data*df[Columns[item]]
        df[Name]=Data*1
    return df


#--------Example-------------------------
#1. Create a dataframe
Sample=np.ones((100,4))
for i in range(Sample.shape[1]):
    Sample[:,i]=i+1.0

df=pd.DataFrame(data=Sample,columns=['[X'+str(x+1)+'_0]' for x in range(Sample.shape[1])])

#2. use function to add combinations to dataframe.
#The second input is for number of parameters per group
df2=AddProductTerms(df,2)

