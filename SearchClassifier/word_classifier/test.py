import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from SearchClassifier.word_classifier.data_loader import random_training_example, training_example
from SearchClassifier.word_classifier.predict import evaluate
from SearchClassifier.word_classifier.utils import category_from_output


def test_with_plot(model, dataset, categories):
    n_categories = len(categories)
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        try:
            category, line, category_tensor, line_tensor = training_example(categories, dataset, i)
            output = evaluate(model, line_tensor)
            guess, guess_i = category_from_output(categories, output)
            category_i = categories.index(category)
            confusion[category_i][guess_i] += 1
        except:
            print(i, "omitted")

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()

    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    # ax.set_xticklabels([''] + categories, rotation=90)
    # ax.set_yticklabels([''] + categories)

    # Force label at every tick
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()
