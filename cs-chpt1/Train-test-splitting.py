from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_csv("assets/north_korea_missile_test_database.csv")
y = df["Missile Name"]
X = df.drop("Missile Name", axis=1)

#train test set
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=31
)

#validation-set do compare and obtain accurate indicator
X_train, X_val, y_train, y_val = train_test_split(
 X_train, y_train, test_size=0.25, random_state=31
)


print (len(X_train))
print (len(y_train))
print (len(X_val))
print (len(y_val))
print (len(X_test))
print (len(y_test))