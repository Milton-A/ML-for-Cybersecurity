import pandas as pd
kdd_df = pd.read_csv("assets/kddcup_dataset.csv", index_col=None)

# Examine the proportion of types of traffic:
y = kdd_df["label"].values
from collections import Counter
Counter(y).most_common()

#Convert all non-normal observations into a single class

def label_anomalous(text):
     """Binarize target labels into normal or anomalous."""
     if text == "normal":
        return 0
     else:
        return 1

kdd_df["label"] = kdd_df["label"].apply(label_anomalous)

#Obtain the ratio of anomalies to normal observations. This is the contamination
#parameter that will be used in our isolation forest:

y = kdd_df["label"].values
counts = Counter(y).most_common()
contamination_parameter = counts[1][1] / (counts[0][1] +
counts[1][1])

#Convert all categorical features into numerical form:
from sklearn.preprocessing import LabelEncoder
encodings_dictionary = dict()
for c in kdd_df.columns:
     if kdd_df[c].dtype == "object":
         encodings_dictionary[c] = LabelEncoder()
         kdd_df[c] = encodings_dictionary[c].fit_transform(kdd_df[c])

#Split the dataset into normal and abnormal observations:

kdd_df_normal = kdd_df[kdd_df["label"] == 0]
kdd_df_abnormal = kdd_df[kdd_df["label"] == 1]
y_normal = kdd_df_normal.pop("label").values
x_normal = kdd_df_normal.values
y_anomaly = kdd_df_abnormal.pop("label").values
x_anomaly = kdd_df_abnormal.values

#Train-test split the dataset:
from sklearn.model_selection import  train_test_split

x_normal_train, x_normal_test, y_normal_train, y_normal_test = train_test_split(x_normal, y_normal, test_size=0.3, random_state=11)
x_anomaly_train, x_anomaly_test, y_anomaly_train, y_anomaly_test = train_test_split(x_anomaly, y_anomaly, test_size=0.3, random_state=11)

import numpy as np

x_train = np.concatenate((x_normal_train, x_anomaly_train))
y_train = np.concatenate((y_normal_train, y_anomaly_train))
x_test = np.concatenate((x_normal_test, x_anomaly_test))
y_test = np.concatenate((y_normal_test, y_anomaly_test))

#Instantiate and train an isolation forest classifier:

from sklearn.ensemble import IsolationForest

IF = IsolationForest(contamination=contamination_parameter)
IF.fit(x_train)

#Score the classifier on normal and anomalous observations:

decisionScores_train_normal = IF.decision_function(x_normal_train)
decisionScores_train_anomaly = IF.decision_function(x_anomaly_train)

#Plot the scores for the normal set:

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
_ = plt.hist(decisionScores_train_normal, bins=50)
plt.show()

plt.figure(figsize=(20, 10))
_ = plt.hist(decisionScores_train_anomaly, bins=50)
plt.show()

#Select a cut-off so as to separate out the anomalies from the normal observations:
cutoff = 0

#Examine this cut-off on the test set:
print(Counter(y_test))
print(Counter(y_test[cutoff > IF.decision_function(x_test)]))

