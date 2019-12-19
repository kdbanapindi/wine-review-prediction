# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:16:29 2019

@author: krish
"""
import pandas as pd
import os
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

from gensim.models import KeyedVectors
# load the Stanford GloVe model
filename = 'glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

import re
import time

os.chdir('C:/Users/krish/OneDrive/Krishna/Coursework/STAT542/Ind_project/wine-reviews')

inp_data=pd.read_csv('inp_data.csv')
proc_data=inp_data.drop_duplicates(subset='description', keep="first")

proc_data=proc_data.reset_index(drop=True)

X = proc_data['description']

result_array=[]

t1=time.time()
for i in range(1,len(X)):
    x_strip=re.sub(r'[^\w\s]','',X[i])
    list_token=x_strip.split()
    token_vec=np.zeros(100)

    for j in range(1,len(list_token)):
        
        if list_token[j] in model.vocab:
            
            token_vec=token_vec+model.get_vector(list_token[j])
            
        else:
            
            token_vec=token_vec
        
    token_vec=token_vec.reshape(-1,1)
    
    token_vec=token_vec.squeeze()
    
    result_array.append(token_vec)
    
    if i%1000==999:
        print(i+1)
    
t2=time.time()

result_df=pd.DataFrame(result_array)    

result_df.to_csv('word2vec.csv')        


            
    
    