import time
import datetime
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from matplotlib import pyplot as plt
from sklearn import metrics
import joblib 
import pickle
import os

starting_time = datetime.datetime.now()
print("Beginning script, time:", starting_time)

PATH_DATASET = "data/kn8_cleaned_norow_encoding.csv"
N_BATCHES = 67
BATCH_SIZE = 100


os.mkdir("api/models/")
print("Preparing data...", time.asctime())
data = pd.read_csv(PATH_DATASET, sep=';', encoding='cp437', error_bad_lines=False)
data = data.dropna()
print(data.head())
classes_list = data.kn8.value_counts()[0: (BATCH_SIZE * N_BATCHES)].axes[0].to_list()
# classes = data.kn8.value_counts()[0:49].axes[0]
condition = data['kn8'].isin(classes_list)
data = data[condition]
def label_row(row, classes_list, batch_size):
    return int(classes_list.index(row['kn8']) / BATCH_SIZE)
data['label'] = data.apply(lambda row: label_row(row, classes_list, BATCH_SIZE), axis=1)
print("Data prepared")
print(data.head())

X = data['product']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print("Data splitted, big training")
# Vectorizing dataset
# Vectorizer
print("Vectorizing big training")
vect = CountVectorizer()
# fit and transform training data
vect.fit(X_train)
print("Saving vectorizer..")
pickle.dump(vect, open("api/models/countVectorizer.pickel", "wb"))
X_train_dtm = vect.transform(X_train)
print(X_train_dtm.shape)
# fit and transform test data
X_test_dtm = vect.transform(X_test)
print("text shape: ", X_test_dtm.shape)
nb = ComplementNB()
#%%
# 3. train the model
# using X_train_dtm (timing it with an IPython "ma         5ygic command")
print("training data big training", time.asctime())
nb.fit(X_train_dtm, y_train)

#%%
# 4. make class predictions for X_test_dtm
print("predicting ", time.asctime(), "big training")
y_pred_class = nb.predict(X_test_dtm)
print(metrics.accuracy_score(y_test, y_pred_class))

#%%
# print the confusion matrix
print("Confusion matrix", time.asctime(), "big training")
cm = metrics.confusion_matrix(y_test, y_pred_class)
df_cm = pd.DataFrame(cm)
# plt.figure(figsize=(10,7))
df_cm_norm = (df_cm-df_cm.mean()) / df_cm.std()
max_df = df_cm_norm.max().max()
# Normalice anr put in a range
df_cm_norm *= 10/max_df
sn.set(font_scale=0.5)  # for label size
sn.heatmap(df_cm_norm, annot=False)  # font size
plt.show()

print('saving big training model..', time.asctime())
joblib.dump(nb, 'api/models/model_big.pk1')

models = []

for i in range(N_BATCHES):
    condition = data['kn8'].isin(classes_list[i * BATCH_SIZE : ( i + 1) * BATCH_SIZE - 1])
    filtered_data = data[condition]
    X = filtered_data['product']
    y = filtered_data['kn8']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    print("Data splitted", "iteration", i)
    # Vectorizing dataset
    # Vectorizer
    print("vectorizing", "iteration", i)
    # fit and transform test data
    X_train_dtm = vect.transform(X_train)
    X_test_dtm = vect.transform(X_test)
    print("test shape: ", X_test_dtm.shape)
    nb = ComplementNB()
    #%%
    # 3. train the model
    # using X_train_dtm (timing it with an IPython "magic command")
    print("training data ", time.asctime(), "iteration", i)
    nb.fit(X_train_dtm, y_train)

    #%%
    # 4. make class predictions for X_test_dtm
    print("predicting ", time.asctime(), "iteration", i)
    y_pred_class = nb.predict(X_test_dtm)
    print(metrics.accuracy_score(y_test, y_pred_class))

    #%%
    joblib.dump(nb, 'api/models/model_' + str(i) + '.pk1')


current_time = datetime.datetime.now()
time_difference = current_time - starting_time
print('time transcurred:', time_difference)
print('Script ended, bye bye!')