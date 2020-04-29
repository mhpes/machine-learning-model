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
from torch import nn

from SearchClassifier.word_classifier.data_loader import randomTrainingExample, split_dataset
from SearchClassifier.word_classifier.predict import evaluate, predict
from SearchClassifier.word_classifier.rnn import RNN
from SearchClassifier.word_classifier.train import train
from SearchClassifier.word_classifier.utils import unicode_to_ascii, letter_to_tensor, line_to_tensor, n_letters

TRAINING_PERCENT = 70
TEST_PERCENT = 20
VIEW_PERCENT = 10
LEARNING_RATE = 0.005  # If you set this too high, it might explode. If too low, it might not learn
EPOCHS = 2000
PRINT_EVERY = 100
PLOT_EVERY = 100
N_HIDDEN = 128


def findFiles(path): return glob.glob(path)


print(findFiles('data/products/*.txt'))

print(unicode_to_ascii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


for filename in findFiles('data/products/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

training_dataset, test_dataset, view_dataset = split_dataset(category_lines, TRAINING_PERCENT, TEST_PERCENT, VIEW_PERCENT)

rnn = RNN(n_letters, N_HIDDEN, n_categories)
inputChar = letter_to_tensor('A')

hidden = torch.zeros(1, N_HIDDEN)

output, next_hidden = rnn(inputChar, hidden)


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


print(categoryFromOutput(output))


for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample(all_categories, training_dataset)
    print('category =', category, '/ line =', line)

# NLLLoss() is good because the last layer of the RNN is nn.LogSoftmax
# I have to see the different criterion and activation functions
criterion = nn.NLLLoss()

# each loop of training
#   1) Create input and target tensors
#   2) Create zeroed initial hidden state
#   3) Read each letter in and keep hidden state for next letter
#   4) Compare final output to target
#   5) Back-propagate
#   6) Return to output and loss
# Keep track of losses for plotting
current_loss = 0
all_losses = []


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()

for iter in range(1, EPOCHS + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample(all_categories, training_dataset)
    output, loss = train(rnn, category_tensor, line_tensor, criterion, LEARNING_RATE)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % PRINT_EVERY == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' %
              (iter, iter / EPOCHS * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % PLOT_EVERY == 0:
        all_losses.append(current_loss / PLOT_EVERY)
        current_loss = 0


plt.figure()
plt.plot(all_losses)
# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000


# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample(all_categories, test_dataset)
    output = evaluate(rnn, line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()

ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()


for key in view_dataset.keys():
    for value in view_dataset[key]:
        predict(rnn, value, all_categories, 3)
