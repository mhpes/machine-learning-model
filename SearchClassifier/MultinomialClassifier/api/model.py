import joblib 
import pickle
import time
import datetime
import pandas as pd
# from flask_csv import send_csv
N_BATCHES = 67
BATCH_SIZE = 100
HEROKU_PATH = "SearchClassifier/MultinomialClassifier/api/"
PATH_BIG_MODEL = 'models/model_big.pk1'
PATH_MODELS = 'models/model_'

PATH_MODELS_EXTENSION = '.pk1'
PATH_VECTORIZER = 'models/countVectorizer.pickel'
TMP_FILES = "tmp/"

BIG_MODEL = joblib.load('models/model_big.pk1')
VECT = pickle.load(open("models/countVectorizer.pickel", "rb"))




def predict_csv(filename, data):
    print("Loading data...", time.asctime())
    # features = ["kn8", "product"]
    # data = pd.read_csv(PATH_INPUTS + csv_input, sep=';', encoding='cp437', error_bad_lines=False)
    # data = data.dropna()
    X = data['product']

    X_dtm = VECT.transform(X)

    print("Data loaded", time.asctime())    
    index = BIG_MODEL.predict(X_dtm)
    model_predicts = []
    for i in range(N_BATCHES):
        model_from_file = joblib.load(PATH_MODELS+str(i)+PATH_MODELS_EXTENSION)
        model_predicts.append(model_from_file.predict(X_dtm))
        porcentaje = (i+1) / N_BATCHES * 100
        print("progress: {:.2f}%".format(porcentaje))
    predicts = []
    for i in range(index.shape[0]):
        result = model_predicts[int(index[i])][i]
        predicts.append(result)
    data["predict"] = predicts
    print("Predicted", time.asctime())
    data.to_csv(TMP_FILES + filename, sep=";", index=False)
    return filename

def predict_product(product):
    product_dict = { "0" : product }
    product_serie = pd.Series(product_dict)
    product_dtm = VECT.transform(product_serie)
    index = BIG_MODEL.predict(product_dtm)
    result = joblib.load(PATH_MODELS+str(index[0])+PATH_MODELS_EXTENSION).predict(product_dtm)
    return str(result[0])