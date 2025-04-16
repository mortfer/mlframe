import numpy as np
import scipy

def WAPE(y_true, y_pred, epsilon=1e-9):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    WAPE = 100*np.sum(np.abs(y_true - y_pred))/(np.sum(np.abs(y_true))+epsilon)
    return WAPE

def MAPE(y_true, y_pred, min_y_true=None, epsilon=1e-9):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if min_y_true is not None:
        bool_mask = (y_true > min_y_true)
        y_true = y_true[bool_mask]
        y_pred = y_pred[bool_mask]
    APE = 100 * np.abs(y_true - y_pred)/(y_true + epsilon)
    return np.mean(APE)

def MAE(y_true, y_pred, epsilon=1e-9):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return np.mean(np.abs(y_true - y_pred))

def APE_percentile(y_true, y_pred, percentile=0.5, min_y_true=None, epsilon=1e-9):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if min_y_true is not None:
        bool_mask = (y_true > min_y_true)
        y_true = y_true[bool_mask]
        y_pred = y_pred[bool_mask]
    APE = 100 * np.abs(y_true - y_pred)/(y_true + epsilon)
    return np.quantile(APE, percentile)

######
#CRPS#
######
def standard_normal_distr(x):
    return 1 / np.sqrt(2*np.pi) * np.exp(-x**2 / 2)

def normal_crps(mus: np.ndarray, sigmas: np.ndarray, y: np.ndarray):
    
    F = scipy.stats.norm.cdf
    epsilon = 1e-7
    x = (y-mus)/(sigmas+epsilon)
    crps = x*(2*F(x)-1) + 2*standard_normal_distr(x) - 1/np.sqrt(np.pi)
    return sigmas*crps

def mixnormal_crps(mus:np.ndarray, sigmas:np.ndarray, pis:np.ndarray, y:np.ndarray):
    #https://krisvillez.gitlab.io/profile/pdf/KV_T004_v1.0.pdf
   
    assert mus.shape == sigmas.shape, "mu and sigma must have the same shape"
    assert mus.shape == pis.shape, "mu and pi must have the same shape"
    y = y.reshape(-1,1)
    assert mus.shape[0] == y.shape[0], "sigmas and y must have the same length"
    
    Phi = scipy.stats.norm.cdf
    phi = standard_normal_distr
    A = lambda mu, sigma: mu*(2*Phi(mu/sigma)-1)+2*sigma*phi(mu/sigma)
    
    mus_i, sigmas_i, pis_i = mus[:, :, np.newaxis], sigmas[:, :, np.newaxis], pis[:, :, np.newaxis]
    mus_j, sigmas_j, pis_j = mus[:, np.newaxis, :], sigmas[:, np.newaxis, :], pis[:, np.newaxis, :]
    
    crps = (pis*A(y-mus, sigmas)).sum(axis=1) - 1/2*(pis_j*pis_i*A(mus_i-mus_j, np.sqrt(sigmas_i**2+sigmas_j**2))).sum(axis=1).sum(axis=1)
    return crps

