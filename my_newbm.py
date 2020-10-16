# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 08:59:06 2020

@author: Grey
"""
#Boltzmann machines
"""##Importing the libraries"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

"""## Importing the dataset"""

# We won't be using this dataset.
movies = pd.read_csv('C:\Boltzmann_Machines/ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('C:\Boltzmann_Machines/ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('C:\Boltzmann_Machines/ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

"""## Preparing the training set and the test set"""

training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

"""## Getting the number of users and movies"""

nb_users = int(max(max(training_set[:, 0], ), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1], ), max(test_set[:, 1])))

"""## Converting the data into an array with users in lines and movies in columns"""

def convert(data):
  new_data = []
  for id_users in range(1, nb_users + 1):
    id_movies = data[:, 1] [data[:, 0] == id_users]
    id_ratings = data[:, 2] [data[:, 0] == id_users]
    ratings = np.zeros(nb_movies)
    ratings[id_movies - 1] = id_ratings
    new_data.append(list(ratings))  
  return new_data
training_set = convert(training_set)
test_set = convert(test_set)

"""## Converting the data into Torch tensors"""

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#converting the ratings into binary ratings 1 (liked) or (not liked)
training_set[training_set==0] = -1
training_set[training_set==1] = 0
training_set[training_set==2] = 0
training_set[training_set>=3] = 1
test_set[test_set==0] = -1
test_set[test_set==1] = 0
test_set[test_set==2] = 0
test_set[test_set>=3] = 1

#creating the architecture of the neural networks
#created batch of hidden notes,visible notes and weights
class RBM():
    def __init__(self, nh, nv):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)