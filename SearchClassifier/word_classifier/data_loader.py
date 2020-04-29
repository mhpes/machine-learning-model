import random
import torch

from SearchClassifier.word_classifier.utils import line_to_tensor


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


def random_training_example(categories, dataset):
    category = random_choice(categories)
    line = random_choice(dataset[category])
    category_tensor = torch.tensor([categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


def split_list(l, training, test, view):
    training_list = []
    test_list = []
    view_list = []
    for i in range(0, training):
        index = random.randint(0, len(l) - 1)
        training_list.append(l[index])
        del l[index]

    for i in range(0, test):
        index = random.randint(0, len(l) - 1)
        test_list.append(l[index])
        del l[index]

    for i in range(0, min(view, len(l) - 1)):  # If there are decimals this max allow us to get the items left
        index = random.randint(0, len(l) - 1)
        view_list.append(l[index])
        del l[index]

    return training_list, test_list, view_list


def split_dataset(category_lines, training, test, view):
    training_dict = {}
    test_dict = {}
    view_dict = {}
    for category in category_lines.keys():
        size = len(category_lines[category])
        training_list, test_list, view_list = split_list(category_lines[category],
                                                         int(size * training / 100), int(size * test / 100), int(size * view / 100))
        training_dict[category] = training_list
        test_dict[category] = test_list
        view_dict[category] = view_list

    return training_dict, test_dict, view_dict
