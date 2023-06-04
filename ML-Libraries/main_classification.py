import pandas as pd
from SL.KNN import KNN
from SL.LogisticRegress import LogisticRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv('https://raw.githubusercontent.com/pvannyamelia/dsc_kuliah/main/data_mining/sources/heart_failure_clinical_records_dataset.csv')

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

X_train, X_test, Y_train,  Y_test = tts(X, Y, test_size=0.25, random_state=20)

train_index = list(X_train.index)
test_index = list(X_test.index)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

X_train = pd.DataFrame(X_train, index=train_index)
X_test = pd.DataFrame(X_test, index=test_index)

# model = KNN(X = X_train, Y = Y_train)
# Y_pred = model.predict(type='C', pred=X_test, K=5)
model = LogisticRegression(X_train, Y_train)
Y_pred = model.predict(X_test)

score = accuracy_score(Y_test, Y_pred)
cm = confusion_matrix(Y_test, Y_pred)

print(cm)
print(score)