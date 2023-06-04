import numpy as np
import math as m
import pandas as pd
from sklearn.model_selection import train_test_split as tts

class KNN :
    def __init__(self, **var) :
        self.X = var['X']
        self.Y = var['Y']
    
    """ Input K : K dalam K nearest neaighbour (integer), 
    pred : x yang mau diprediksi  (dataframe)
    type : type regression atau classification (C atau R)"""
    def predict(self, **var):
        def calculate_distance(x, pred):
            result = 0
            for i in range(len(x)):
                result += (x[i] - pred[i]) * (x[i] - pred[i])
            result = m.sqrt(result)
            return result
        def get_result(y, indexes, type):
            res = []
            for index in indexes:
               res.append(y.loc[index]) 
            if type == "C":
                set_res = set(res)
                count = 0
                mode = ""
                for r in set_res:
                    num = res.count(r)
                    if num > count:
                        count = num
                        mode = r
                return mode
            if type == "R":
                mean = np.mean(res)
                return mean
        distance = dict()
        x_pred = var['pred']
        K = int(var['K'])
        y_pred = {"Prediction" : []}
        for index, row in x_pred.iterrows():
            distance[index] = {"Index" : [], "Distance" : []}
            for index2, row2 in self.X.iterrows():
               distance[index]['Distance'].append(calculate_distance(row2, row))
               distance[index]['Index'].append(index2)
            temp_res = [res for _, res in sorted(zip(distance[index]['Distance'], distance[index]['Index']))]
            temp_res = temp_res[: K]
            y_pred['Prediction'].append(get_result(self.Y, temp_res, var['type']))
        y_pred = pd.DataFrame(y_pred)
        return y_pred
            

        