import pandas as pd
data = pd.read_csv("assets/file_pe_headers.csv", sep=",")
X = data.drop(["Name", "Malware"], axis=1).to_numpy()


print(X)
print()

#standardize

from sklearn.preprocessing import StandardScaler
X_standardized = StandardScaler().fit_transform(X)

print(X_standardized)