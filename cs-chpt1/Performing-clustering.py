#Start by importing and plotting the dataset:

import pandas as pd
import plotly.express as px
df = pd.read_csv("assets/file_pe_headers.csv", sep=",")
fig = px.scatter_3d(
 df,
 x="SuspiciousImportFunctions",
 y="SectionsLength",
 z="SuspiciousNameSection",
 color="Malware",
)
#fig.show()


#Extract the features and target labels:

y = df["Malware"]
X = df.drop(["Name", "Malware"], axis=1).to_numpy()


#Em seguida, importe o módulo de clustering do scikit-learn e ajuste um modelo K-means com dois clusters para os dados:

from sklearn.cluster import KMeans
estimator = KMeans(n_clusters=len(set(y)))
estimator.fit(X)

#Predict the cluster using our trained algorithm:
y_pred = estimator.predict(X)
df["pred"] = y_pred
df["pred"] = df["pred"].astype("category")

#To see how the algorithm did, plot the algorithm's clusters:

fig = px.scatter_3d(
 df,
 x="SuspiciousImportFunctions",
 y="SectionsLength",
 z="SuspiciousNameSection",
 color="pred",
)
#fig.show()

print(y_pred)