import pandas as pd
df = pd.read_csv("assets/file_pe_headers.csv", sep=",")
y = df["Malware"]
X = df.drop(["Name", "Malware"], axis=1).to_numpy()

#Next, train-test-split a dataset:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.3)


#Create one instance of an XGBoost model and train it on the training set:

from xgboost import XGBClassifier
XGB_model_instance = XGBClassifier()
XGB_model_instance.fit(X_train, y_train)

#Finally, assess its performance on the testing set:

from sklearn.metrics import accuracy_score
y_test_pred = XGB_model_instance.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy: %.2f%%" % (accuracy * 100))