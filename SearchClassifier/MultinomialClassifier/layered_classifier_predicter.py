import time
import datetime
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plt
from sklearn import metrics
import joblib 
import pickle
import resource
import platform
import sys

starting_time = datetime.datetime.now()
print("Beginning script, time:", starting_time)

PATH_DATASET = "data/kn8_cleaned_norow_encoding.csv"
N_BATCHES = 67
BATCH_SIZE = 100


print("Preparing data...", time.asctime())
features = ["kn8", "product"]
data = pd.read_csv(PATH_DATASET, sep=';', encoding='cp437', error_bad_lines=False)
data = data.dropna()
classes_list = data.kn8.value_counts()[0: (BATCH_SIZE * N_BATCHES)].axes[0].to_list()
# classes = data.kn8.value_counts()[0:49].axes[0]
condition = data['kn8'].isin(classes_list)
data = data[condition]
# def label_row(row, classes_list, batch_size):
#     return int(classes_list.index(row['kn8']) / BATCH_SIZE)
# data['label'] = data.apply(lambda row: label_row(row, classes_list, BATCH_SIZE), axis=1)
print("Data prepared")


print("Splitting data", time.asctime())
X = data['product']
y = data['kn8']
#print(X)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print("Vectorizing big training", time.asctime())
#vect = HashingVectorizer()
# fit and transform test data
#print(X_train.shape)
# X_train_dtm = vect.fit(X_train)
print(X_test.head())
print(y_test.head())
vect = pickle.load(open("api/models/countVectorizer.pickel", "rb"))
X_test_dtm = vect.transform(X_test)
print(X_test_dtm.shape)

def predict(data):
    big_model = joblib.load('api/models/model_big.pk1')
    index = big_model.predict(data)
    del big_model
    model_predicts = []
    for i in range(N_BATCHES):
        model_from_file = joblib.load('api/models/model_'+str(i)+'.pk1')
        model_predicts.append(model_from_file.predict(data))
        print("batch: ", i)
    predicts = []
    for i in range(index.shape[0]):
        result = model_predicts[int(index[i])][i]
        predicts.append(result)
    return predicts

print("Predicting...", time.asctime())
y_pred_class = predict(X_test_dtm)
print(metrics.accuracy_score(y_test, y_pred_class))

# print the confusion matrix
print("Confusion matrix", time.asctime(), "big training")
cm = metrics.confusion_matrix(y_test, y_pred_class)
cmap = sn.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True)
print(cm.shape)
df_cm = pd.DataFrame(cm)
# plt.figure(figsize=(10,7))

# Normalice and put in a range
df_cm_norm = (df_cm-df_cm.mean()) / df_cm.std()
max_df = df_cm_norm.max().max()
df_cm_norm *= 10/max_df
sn.set(font_scale=0.5)  # for label size
sn.heatmap(df_cm_norm, annot=False, cmap=cmap)  # font size
plt.show()
current_time = datetime.datetime.now()
time_difference = current_time - starting_time
print('time transcurred:', time_difference)
print('Script ended, bye bye!')