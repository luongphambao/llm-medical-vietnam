import numpy as np 

#RMSE
def rmse(y_true, y_pred):
    #MSE FOR np.log(y_true) and np.log(y_pred)
    return (np.mean((np.log(y_true) - np.log(y_pred))**2))
print(rmse([106,54.2, 10], [100, 50, 12])) #0.0