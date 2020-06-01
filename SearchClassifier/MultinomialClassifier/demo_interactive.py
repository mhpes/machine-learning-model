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

def predict_csv(csv_input, csv_output):
    print("Loading data...", time.asctime())
    features = ["kn8", "product"]
    data = pd.read_csv(PATH_INPUTS + csv_input, sep=';', encoding='cp437', error_bad_lines=False)
    data = data.dropna()
    X = data['product']

    X_dtm = VECT.transform(X)

    print("Data loaded", time.asctime())    
    big_model = joblib.load('models/model_big.pk1')
    index = big_model.predict(X_dtm)
    del big_model
    model_predicts = []
    for i in range(N_BATCHES):
        model_from_file = joblib.load('models/model_'+str(i)+'.pk1')
        model_predicts.append(model_from_file.predict(X_dtm))
        porcentaje = (i+1) / N_BATCHES * 100
        print("progress: {:.2f}%".format(porcentaje))
    predicts = []
    for i in range(index.shape[0]):
        result = model_predicts[int(index[i])][i]
        predicts.append(result)
    data["predict"] = predicts
    data.to_csv("outputs/" + csv_output, sep=";", index=False)
    return predicts

def predict_product(product):
    product_dict = { "0" : product }
    product_serie = pd.Series(product_dict)
    product_dtm = VECT.transform(product_serie)
    big_model = joblib.load('models/model_big.pk1')
    index = big_model.predict(product_dtm)
    result = joblib.load('models/model_'+str(index[0])+'.pk1').predict(product_dtm)
    return result[0]


starting_time = datetime.datetime.now()
print("Beginning script, time:", starting_time)
VECT = pickle.load(open("models/countVectorizer.pickel", "rb"))

EXEC_MODE = 0
while EXEC_MODE != 3:
    try:
        EXEC_MODE = int(input("Insert exec mode: \n 1) Insert product \n 2) Insert csv name \n 3) Exit \n"))
        if EXEC_MODE == 1:
            product = input("Insert product: ")
            print("Predicting..", time.asctime())
            prediction = str(predict_product(product))
            print("The product is of class: " + prediction)
        elif EXEC_MODE == 2:
            csv_input = input("Insert csv input name: ")
            csv_output = input("Insert csv output name: ")
            print("Predicting..", time.asctime())
            predict_csv(csv_input, csv_output)
        elif EXEC_MODE == 3:
            print("Exiting...", time.asctime())
            current_time = datetime.datetime.now()
            time_difference = current_time - starting_time
            print('time transcurred:', time_difference)
    except (ValueError):
        print("Insert a number pls!!")
    except (FileNotFoundError):
        print("Insert a valid input document")
    
