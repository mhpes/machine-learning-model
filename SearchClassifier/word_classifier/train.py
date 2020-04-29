import math
import time

from SearchClassifier.word_classifier.data_loader import random_training_example
from SearchClassifier.word_classifier.utils import category_from_output


def train(model, category_tensor, line_tensor, criterion, learning_rate):
    hidden = model.init_hidden()

    model.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in model.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


def train_loop(model, criterion, categories, dataset, epochs,  learning_rate, print_every, plot_every):
    current_loss = 0
    all_losses = []

    # optimizer = optim.SGD(rnn.parameters(), lr=LEARNING_RATE)

    def time_since(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    start = time.time()

    for iter in range(1, epochs + 1):
        model.zero_grad()
        category, line, category_tensor, line_tensor = random_training_example(categories, dataset)
        output, loss = train(model, category_tensor, line_tensor, criterion, learning_rate)
        current_loss += loss
        # optimizer.step()
        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = category_from_output(categories, output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' %
                  (iter, iter / epochs * 100, time_since(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    return model, all_losses
