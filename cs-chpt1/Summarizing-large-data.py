#Summarizing large data using principal component analysis

from sklearn.decomposition import PCA
import pandas as pd


data = pd.read_csv("assets/file_pe_headers.csv", sep=",")
X = data.drop(["Name", "Malware"], axis=1).to_numpy()

#Standardize the dataset, as is necessary before applying PCA:

from sklearn.preprocessing import StandardScaler
X_standardized = StandardScaler().fit_transform(X)

#Instantiate a PCA instance and use it to reduce the dimensionality of our data:

pca = PCA()
pca.fit_transform(X_standardized)

#Assess the effectiveness of your dimensionality reduction:
print(pca.explained_variance_ratio_)

sum(pca.explained_variance_ratio_[0:40])