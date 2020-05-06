#%% md

## Imports

#%%

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import random
import unicodedata
import string
import torch
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch import nn, optim
from SearchClassifier.word_classifier.data_loader import random_training_example, split_dataset, load_dataset, \
    split_dataset_reduced
from SearchClassifier.word_classifier.mutiLayerRNN import MultilayerRnn
from SearchClassifier.word_classifier.predict import evaluate, predict
from SearchClassifier.word_classifier.rnn import RNN
from SearchClassifier.word_classifier.test import test_with_plot
from SearchClassifier.word_classifier.train import train, train_loop
from SearchClassifier.word_classifier.utils import unicode_to_ascii, letter_to_tensor, line_to_tensor, n_letters, \
    category_from_output, load_checkpoint, save_checkpoint


## Global variables
#%%

TRAINING_PERCENT = 70
TEST_PERCENT = 20
VIEW_PERCENT = 10
LEARNING_RATE = 0.005  # If you set this too high, it might explode. If too low, it might not learn
EPOCHS = 5000
PRINT_EVERY = 250
PLOT_EVERY = 250
N_HIDDEN = 1500
N_HIDDEN_2 = 512
EXEC_MODE = 1
REDUCE = 15
PATH_MODEL_1 = "models/model.pt"
PATH_MODEL_2 = "models/model2.pt"
PATH_DATASET = "data/kn8_defined_cleaned_data.csv"


### Configuration

#%%

dataset = load_dataset(PATH_DATASET)

#%%

print(dataset.keys())

#%%

training_dataset, test_dataset, view_dataset = split_dataset_reduced(dataset, TRAINING_PERCENT, TEST_PERCENT, VIEW_PERCENT, REDUCE)
all_categories = list(training_dataset.keys())
print(len(all_categories))

## Creating RNN

#%%
if EXEC_MODE == 1:
    rnn = MultilayerRnn(n_letters, N_HIDDEN, len(all_categories))
elif EXEC_MODE == 2:
    rnn = RNN(n_letters, N_HIDDEN, len(all_categories))
    rnn2 = MultilayerRnn(n_letters, N_HIDDEN, len(all_categories))
else:
    print("Loading network")
    rnn = RNN(n_letters, N_HIDDEN, len(all_categories))
    rnn = load_checkpoint(rnn, PATH_MODEL_1)
    #rnn.eval()
inputChar = letter_to_tensor('A')

hidden = torch.zeros(1, N_HIDDEN)

rnn

### Getting output

#%%
output, next_hidden = rnn(inputChar, hidden)
category_from_output(all_categories, output)


### Showing samples
#%%

for i in range(10):
    category, line, category_tensor, line_tensor = random_training_example(all_categories, training_dataset)
    print('category =', category, '/ line =', line)


#%%

if EXEC_MODE == 1 or EXEC_MODE == 2:
    criterion = nn.NLLLoss()



#%%

if EXEC_MODE == 1:
    print("Training network")
    rnn, all_losses = train_loop(rnn, criterion, all_categories, training_dataset, EPOCHS, LEARNING_RATE, PRINT_EVERY, PLOT_EVERY)
if EXEC_MODE == 2:
    print("Training first network")
    rnn, all_losses = train_loop(rnn, criterion, all_categories, training_dataset, EPOCHS, LEARNING_RATE, PRINT_EVERY, PLOT_EVERY)
    print("Training second network")
    rnn2, all_losses2 = train_loop(rnn2, criterion, all_categories, training_dataset, EPOCHS, LEARNING_RATE, PRINT_EVERY, PLOT_EVERY)


#%%

if EXEC_MODE == 1 or EXEC_MODE == 2:
    plt.figure()
    plt.plot(all_losses)
if EXEC_MODE == 2:
    plt.figure()
    plt.plot(all_losses2)

#%%

test_with_plot(rnn, test_dataset, all_categories)
if EXEC_MODE == 2:
    test_with_plot(rnn2, test_dataset, all_categories)

#%%

save_checkpoint(rnn, PATH_MODEL_1)
if EXEC_MODE == 2:
    save_checkpoint(rnn2, PATH_MODEL_2)
print("saved")

#%% md

### Some predicts

#%%

for key in view_dataset.keys():
    for value in view_dataset[key]:
        predict(rnn, value, all_categories, 3)
