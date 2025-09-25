import numpy as np
from tqdm.auto import tqdm

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def CONSISTENCY(pred):
    consistency = 0
    for i in tqdm(range(len(pred) - 1), desc="Computing Consistency MAE"):
        pred1 = pred[i][1:]
        pred2 = pred[i + 1][:-1]
        consistency += np.mean(np.abs(pred1 - pred2))
    return -np.log(consistency / (len(pred) - 1)) 

def CONSISTENCY_MSE(pred):
    consistency = 0
    for i in tqdm(range(len(pred) - 1), desc="Computing Consistency MSE"):
        pred1 = pred[i][1:]
        pred2 = pred[i + 1][:-1]
        consistency += np.mean((pred1 - pred2) ** 2)
    return -np.log(consistency / (len(pred) - 1))

def TAM(consistency):
    return np.exp(-consistency)

def metric(pred, true):
    print(pred.shape, true.shape)
    print("Computing MAE")
    mae = MAE(pred, true)
    print("Computing MSE")
    mse = MSE(pred, true)
    print("Computing RMSE")
    rmse = RMSE(pred, true)
    print("Computing MAPE")
    mape = MAPE(pred, true)
    print("Computing MSPE")
    mspe = MSPE(pred, true)
    print("Computing RSE")

    return mae, mse, rmse, mape, mspe
