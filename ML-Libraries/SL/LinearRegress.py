import numpy as np
import pandas as pd
import math
import random as rnd
from pyparsing import And
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse

class OLSLinearRegression :
    
    def __init__(self, X, Y, **kwargs):
        self.X = X
        self.Y = Y
        self.learn()
    
    def learn(self):
        # menghitung nilai B dan C secara OLS
        def calculate_C(X, Y, B) :
            X = np.asarray(X)
            Y = np.asarray(Y)
            mean_Y = np.mean(Y)
            mean_X = []
            row = len(X)
            for i in range(row):
                for j in range(len(X[i])):
                    if i == 0:
                        mean_X.append(0.0)
                    mean_X[j] += (X[i][j] / row)
            C = mean_Y
            for i in range(len(mean_X)):
                C -= mean_X[i] * B[i]
            return C
        
        #membuat X dan Y Matrix
        if isinstance(self.X , pd.DataFrame):
            X = self.X.to_numpy(copy=True) 
            X = np.matrix(X)
        else:
            X = np.matrix(self.X)
        if isinstance(self.Y, pd.DataFrame):
            Y = self.Y.to_numpy(copy=True) 
            Y = np.matrix(Y)
        else:
            Y = np.matrix(self.Y)
        
        Y = np.transpose(Y)
        X_t = np.transpose(X)
        # Rumus = B = (Xt . X)^ -1 . Xt. Y
        B = np.dot(np.linalg.pinv(np.dot(X_t, X)), np.dot(X_t, Y))
        # B = np.dot(X_t, X)
        self.B = B
        self.C = calculate_C(X, Y, B)
    
    def predict(self, X_pred):
        if isinstance(X_pred , pd.DataFrame):
            X_pred = X_pred.to_numpy(copy=True) 
            X_pred = np.matrix(X_pred)
        else:
            X_pred = np.matrix(X_pred)
        Y_pred = np.dot(X_pred, self.B)
        Y_pred = np.asarray(Y_pred)
        Y_pred = Y_pred[:, 0]
        for i in range (len(Y_pred)):
            Y_pred[i] += self.C
        return Y_pred
    
    def report(self):
        print("Weight : {0}".format(self.B))
        print("Intercept : {0}".format(self.C))
        print('')

class GDLinearRegression:
    def __init__(self, X, Y, **kwargs):
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
        if "learning_rate" in kwargs.keys():
            self.LR = float(kwargs['learning_rate'])
        else:
            self.LR = 0.01
        if "min_step_size" in kwargs.keys():
            self.min_step_size = float(kwargs['min_step_size'])
        else:
            self.min_step_size = None
        if "max_step" in kwargs.keys():
            self.max_step = float(kwargs['max_step'])
        else:
            self.max_step = None
        if "min_loss" in kwargs.keys():
            self.min_loss= float(kwargs['min_loss'])
        else:
            self.min_loss = None
        self.B = np.random.randn(len(self.X[0]))
        self.C = rnd.random()
        self.learn()
        
        
    def learn(self):
        def loss(y, y_pred):
            n = len(y)
            result = 0.0
            for i in range(n):
                result += (y[i] - y_pred[i])**2
            result /= n
            result = math.sqrt(result)
            return result
        #X merupakan salah satu variabel dalam seluruh data training
        def dlb(y, y_pred, x):
            result = 0
            n = len(y)
            for i in range(n):
                result += x[i] * (y[i] - y_pred[i])
            result *= 2 / n
            return result
        def dlc(y, y_pred):
            result = 0
            n = len(y)
            for i in range(n):
                result += (y[i] - y_pred[i])
            result *= 2 / n
            return result
        def stop_c(step_size, num_of_steps, loss, min_step_size, max_num_of_steps, min_loss):
            if max_num_of_steps is not None:
                if num_of_steps >= max_num_of_steps:
                    return True
            result = 0
            if min_step_size is not None:
                for s in step_size:
                    if s > min_step_size:
                        result = 1
                if result == 0:
                    return True
            if loss < min_loss:
                return True
            return False
        
        sc = False
        epoch = 0
        if self.min_loss is None:
            self.min_loss = 0.1 * np.mean(self.Y)
        while sc == False:
            epoch += 1
            y_pred = self.predict(self.X)
            e_loss = loss(self.Y, y_pred)
            step_size = []
            for i in range (len(self.B)):
                step = self.LR * dlb(self.Y, y_pred, self.X[:, i])
                step_size.append(step)
                self.B[i] += step
            step = self.LR * dlc(self.Y, y_pred)
            step_size.append(step)
            self.C += step
            sc = stop_c(step_size, epoch, e_loss, self.min_step_size, self.max_step, self.min_loss)
        print(epoch)
            
    
    def predict(self, X):
        if isinstance(X , pd.DataFrame) or isinstance(X , pd.Series):
            X = X.to_numpy(copy=True) 
        elif isinstance(X , np.ndarray) == False:
            X = np.array(X)
        y_pred = []
        for var in X:
            pred = 0.0
            for i in range(len(var)):
                #Error
                try:
                    pred += self.B[i] * var[i]
                except Exception as e:
                    print(X)
                    print(var)
                    print(self.B)
                    print(self.B[i])
            pred += self.C
            y_pred.append(pred)
        return y_pred
    def report(self):
        print("Weight : {0}".format(self.B))
        print("Intercept : {0}".format(self.C))
        print('')
            
            
        
            
# df = pd.read_csv('https://raw.githubusercontent.com/pvannyamelia/dsc_kuliah/main/data_science/assets_dsc/Salary_Data.csv')
# df = pd.read_csv('https://raw.githubusercontent.com/pvannyamelia/dsc_kuliah/main/data_science/assets_dsc/real_estate.csv')

# X = df.iloc[:, :-1]
# Y = df.iloc[:, -1]  

# X_train, X_test, Y_train,  Y_test = tts(X, Y, test_size=0.25, random_state=20)

# model = GDLinearRegression(X_train, Y_train)
# model.learn()

# Y_pred = model.predict(X_test)
# print(Y_pred)
# rmse = mse(Y_test, Y_pred, squared=False)
# print(rmse)

        
        
            

