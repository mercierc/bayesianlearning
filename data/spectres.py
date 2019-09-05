import sys
                
import numpy as np
import pandas as pd

import time
from regression_time import BayesianRidge
from regression_time import RVMRegression
from sklearn.linear_model import ARDRegression 
import matplotlib.pyplot as plt
np.random.seed(8)


X = np.genfromtxt('corn_m5_train.csv',delimiter=',')
Xtest = np.genfromtxt('corn_m5_test.csv',delimiter=',')
y = np.genfromtxt('corn_y_train.csv',delimiter=',')
ytest = np.genfromtxt('corn_y_test.csv',delimiter=',')

print('nxp = ',X.shape[0],"x",X.shape[1])
start = time.time()
bayes = BayesianRidge().fit(X,y)
end = time.time()
pred = bayes.predict(X)
predtest = bayes.predict(Xtest)

print("time =", (end-start))


rmsetrain = bayes.score(X, y)
rmsetest = bayes.score(Xtest, ytest)

print("rmsetrain = ", rmsetrain)
print("rmsetest = ", rmsetest)


# Write X and y for the train
import csv
with open('synth_train.csv', mode='w') as file_:
    writer_ = csv.writer(file_, delimiter=',')

    for row in X:
        writer_.writerow(row)
file_.close()
with open('synth_y_train.csv', mode='w') as file_:
    writer_ = csv.writer(file_, delimiter=',')

    for row in y:
        writer_.writerow([row])

        

with open('synth_test.csv', mode='w') as file_:
    writer_ = csv.writer(file_, delimiter=',')

    for row in Xtest:
        writer_.writerow(row)

with open('synth_y_test.csv', mode='w') as file_:
    writer_ = csv.writer(file_, delimiter=',')

    for row in ytest:
        writer_.writerow([row])


