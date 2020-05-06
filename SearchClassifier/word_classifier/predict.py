import torch

from SearchClassifier.word_classifier.utils import line_to_tensor


def evaluate(model, line_tensor):
    hidden = model.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    return output


def to_per_cent(tensor):
    max_value = sum(tensor[0])
    result = [1 - (x / max_value) for x in tensor[0]]
    max_value = sum(result)
    result = [x / max_value for x in result]
    return torch.as_tensor([result])


def predict(model, input_line, all_categories, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(model, line_to_tensor(input_line))
        output = to_per_cent(output.to("cpu").numpy())
        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.10f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])
