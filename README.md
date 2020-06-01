# machine-learning-model

## For running the api

`git clone https://github.com/mhpes/machine-learning-model.git`
`cd machine-learning-model/SearchClassifier/MultinomialClassifier`

### Training models
 1. Change `layered_classifiers_training.py` varname: PATH_DATASET to your relative path to the data on .csv format.
 2. Run `python3 layered_classifiers_training.py` to create models for api consume.

 ### Test models (Optional)
 1.  run `python3 layered_classifier_predicter.py` to check accuracy, test models.

### Run api
 1. `cd api`
 2. `python3 api.py`

Now you can test your models requesting the api. (localhost:5000)
You can see path in api doc.


No original models included due to repository size.