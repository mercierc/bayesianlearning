import sys
                
import numpy as np
import pandas as pd

import time
from regression_time import BayesianRidge
from regression_time import RVMRegression
import matplotlib.pyplot as plt
np.random.seed(8)

N,M = 50,70
Ntest,Mtest = 50,M
s=10
X = np.random.uniform(-10,10,N*M).reshape(N,M)
Xtest = np.random.uniform(-10,10,Ntest*Mtest).reshape(Ntest,Mtest)

coef = np.zeros(M)
coef[:s] = np.ceil(np.random.uniform(-5,5,s))

beta = 40
noise = np.random.normal(0,(1/beta)**0.5,N)
noisetest = np.random.normal(0,(1/beta)**0.5,Ntest)

y = X.dot(coef) + noise
ytest = Xtest.dot(coef) + noisetest


print('nxp = ',X.shape[0],"x",X.shape[1])
start = time.time()
bayes = BayesianRidge(fit_intercept=True, scale=False).fit(X,y)
bayes2 = RVMRegression(fit_intercept=True, scale=False).fit(X,y)
end = time.time()
pred = bayes.predict(X)
predtest = bayes.predict(Xtest)

print("time =", (end-start))


rmsetrain = ((pred-y)**2).mean()**0.5
rmsetest = ((predtest-ytest)**2).mean()**0.5

print("rmsetrain = ",rmsetrain)
print("rmsetest = ",rmsetest)

#plt.plot(y,pred,'.'),plt.plot(ytest,predtest,'.'),plt.show()


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


