# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:39:09 2021

@author: dlimiaperez
"""
import pandas as pd

df1 = pd.read_csv("LiftEquationData.csv")
df2 = df1.drop(df1.columns[[0,1,2,3,4,5]],axis=1)

for i in range (20):
    for j in range (16):
        Input = pd.concat([df2.iloc[:,j]],axis=1)
        InputShifted = Input.shift(periods=i+1)
        df2 = pd.concat([df2,InputShifted],axis=1)
df3 = df2.dropna()
lista = []
for i in range (21):
    for j in range (16):
        a = df2.iloc[:,j].name
        b = '[' + a + '_' + str(i) + ']'
        lista.append (b)

df3.set_axis(lista, axis='columns', inplace=True)
print(df3)