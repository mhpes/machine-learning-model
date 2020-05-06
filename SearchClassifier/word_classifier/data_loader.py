import random
import torch
import pandas as pd
from SearchClassifier.word_classifier.utils import line_to_tensor


def load_dataset(filepath):
    # I read it as dataframe
    data = pd.read_csv(filepath, sep=';', encoding='cp437', error_bad_lines=False)
    print(data.head())
    dataset = {}
    for row in data.itertuples():
        category = row.kn8
        product = row.product
        if category in dataset:
            dataset[category].append(product)
        else:
            dataset[category] = [product]
    return dataset


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


def random_training_example(categories, dataset):
    try:
        category = random_choice(categories)
        line = random_choice(dataset[category])
        category_tensor = torch.tensor([categories.index(category)], dtype=torch.long)
        line_tensor = line_to_tensor(line)
    except:
        print("ERROR", str(category), (line))
        return 0, 0, 0, 0
    return category, line, category_tensor, line_tensor


def training_example(categories, dataset, number):
    category = categories[number % len(categories)]
    line = dataset[category][int((number / len(categories))) % len(dataset[category])]
    category_tensor = torch.tensor([categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


def split_list(l, training, test, view):
    training_list = l[0:training]
    test_list = l[training + 1: training + test]
    view_list = l[training + test + 1: len(l) - 1]
    return training_list, test_list, view_list


def split_dataset(dataset, training, test, view):
    training_dict = {}
    test_dict = {}
    view_dict = {}
    for category in dataset.keys():
        size = len(dataset[category])
        training_list, test_list, view_list = split_list(dataset[category],
                                                         int(size * training / 100), int(size * test / 100),
                                                         int(size * view / 100))
        if view_list:
            training_dict[category] = training_list
            test_dict[category] = test_list
            view_dict[category] = view_list

    return training_dict, test_dict, view_dict


def split_dataset_reduced(dataset, training, test, view, reduce=5):
    training_dict = {}
    test_dict = {}
    view_dict = {}
    for category in dataset.keys():
        size = len(dataset[category])
        training_list, test_list, view_list = split_list(dataset[category],
                                                         int(size * training / 100), int(size * test / 100),
                                                         int(size * view / 100))
        if len(view_list) > reduce:
            training_dict[category] = training_list
            test_dict[category] = test_list
            view_dict[category] = view_list

    return training_dict, test_dict, view_dict
