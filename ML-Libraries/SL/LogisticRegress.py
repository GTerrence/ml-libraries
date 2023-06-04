from SL.LinearRegress import OLSLinearRegression as OLS
# from LinearRegress import OLSLinearRegression as OLS
from sklearn.model_selection import train_test_split as tts
import pandas as pd
import numpy as np
import math 
from sklearn.metrics import confusion_matrix, accuracy_score


class LogisticRegression:
    
    def __init__(self, X, Y):
        if isinstance(X , pd.DataFrame) or isinstance(X , pd.Series):
            self.X = X.to_numpy(copy=True) 
        elif isinstance(X , np.ndarray) == False:
            self.X = np.array(X)
        else:
            self.X = X
        if isinstance(Y , pd.DataFrame) or isinstance(Y , pd.Series):
            self.Y = Y.to_numpy(copy=True) 
        elif isinstance(Y , np.ndarray) == False:
            self.Y = np.array(Y)
        else:
            self.Y = Y
        self.learn()
    
    def learn(self):
        def dlm(y, y_pred):
            result = 0
            n = len(y)
            for i in range(n):
                result += (y[i] - y_pred[i])
            result *= 2 / n
            return result
        def stop_c(step):
            if step < 0.001:
                return True
            return False
        self.Linear = OLS(self.X, self.Y)
        self.margin = 0.5
        y_pred = self.Linear.predict(self.X)
        for i in range(len(y_pred)):
            y_pred[i] = self.sigmoid(y_pred[i])
        epoch = 0
        
        sc = False
        while sc == False:
            epoch += 1
            y_val = []
            for y in y_pred:
                if y >= self.margin:
                    y_val.append(1)
                else:
                    y_val.append(0)
            step = 0.1 * dlm(self.Y, y_val)
            self.margin -= step
            sc = stop_c(abs(step))
        print(epoch) 
        
    def sigmoid(self, val):
        result = 1 / (1 + math.exp(val * -1))
        return result
    
    def predict(self, X_pred) :
        Y_pred = self.Linear.predict(X_pred)
        for i in range(len(Y_pred)):
            Y_val = self.sigmoid(Y_pred[i])
            if Y_val >= self.margin :
                Y_pred[i] = 1
            elif Y_val < self.margin:
                Y_pred[i] = 0
        return Y_pred

# df = pd.read_csv('https://raw.githubusercontent.com/pvannyamelia/dsc_kuliah/main/data_mining/sources/heart_failure_clinical_records_dataset.csv')

# X = df.iloc[:, :-1]
# Y = df.iloc[:, -1]
            
# X_train, X_test, Y_train,  Y_test = tts(X, Y, test_size=0.25, random_state=20)

# model = LogisticRegression(X_train, Y_train)
# Y_pred = model.predict(X_test)

# df_compare = pd.DataFrame({'prediction' : Y_pred, 'actual' : Y_test})        
# print(df_compare)  

# score = accuracy_score(Y_test, Y_pred)
# cm = confusion_matrix(Y_test, Y_pred)

# print(cm)
# print(score)