import time
import datetime
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import metrics
import joblib 
import pickle

N_BATCHES = 67
BATCH_SIZE = 100
PATH_BIG_MODEL = 'models/model_big.pk1'
PATH_MODELS = 'models/model_'
PATH_MODELS_EXTENSION = '.pk1'
PATH_VECTORIZER = 'models/countVectorizer.pickel'
PATH_INPUTS = 'inputs/'
PATH_OUTPUTS = 'outputs/'
CSV_INPUT = 'kn8_definitive.csv'
CSV_OUTPUT = 'kn8_predicts.csv'

starting_time = datetime.datetime.now()
print("Beginning script, time:", starting_time)

print("Loading data...", time.asctime())
features = ["kn8", "product"]
data = pd.read_csv(PATH_INPUTS + CSV_INPUT, sep=';', encoding='cp437', error_bad_lines=False)
data = data.dropna()
print("Data loaded", time.asctime())

print("Splitting data", time.asctime())
X = data['product']
y = data['kn8']
#print(X)
#print(y)

#print(X.head())
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(0.99))
#print(X_test.head())
print("Vectorizing big training", time.asctime())
#vect = HashingVectorizer()
# fit and transform test data
#print(X_train.shape)
# X_train_dtm = vect.fit(X_train)
vect = pickle.load(open("models/countVectorizer.pickel", "rb"))
X_test_dtm = vect.transform(X)
print(X_test_dtm.shape)

def predict(data):
    big_model = joblib.load('models/model_big.pk1')
    index = big_model.predict(data)
    del big_model
    model_predicts = []
    for i in range(N_BATCHES):
        model_from_file = joblib.load('models/model_'+str(i)+'.pk1')
        model_predicts.append(model_from_file.predict(data))
        print("batch: ", i)
    predicts = []
    for i in range(index.shape[0]):
        result = model_predicts[int(index[i])][i]
        predicts.append(result)
    return predicts

print("Predicting...", time.asctime())
y_pred_class = predict(X_test_dtm)
print("predicts", y_pred_class)
print(metrics.accuracy_score(y, y_pred_class))
print(data.shape)
data['predicted'] = y_pred_class
data.to_csv(PATH_OUTPUTS + CSV_OUTPUT, sep=";", index=False)