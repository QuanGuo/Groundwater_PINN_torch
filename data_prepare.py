# data prepare
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from utils import *
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import rcParams

# rcParams['font.family'] = 'serif'
# rcParams['font.size'] = 16

nx = 64
ny = 64

x = np.linspace(0,1,nx)
y = np.linspace(0,1,ny)

Exact = p_analytical(x,y)

u_pred = np.loadtxt('u_pred_torch.txt').reshape((nx,ny))
plot_3D(x,y,u_pred, 'Prediction')
plot_3D(x,y,Exact, 'True')
# plot_map_2d(x, y, Exact-u_pred, (nx,ny))
# compare_true_pred(Exact, u_pred, x, y)
plt.show()

# sigma2 = 1
# lx = 20
# ly = 20

# print(Xc.reshape((nx,ny)))
# print(Yc.reshape((nx,ny)))

# hx = distance_matrix([Xc])
# hy = distance_matrix([Yc])

# Q = sigma2 * np.exp(-np.square(hx/lx) - np.square(hy/ly))

# k=50
# mu = 0.0001
# NR = 100
# y, alpha = pca_realization_generation(Q, k, mu, NR)
# print(alpha.shape)
# print(y.shape)
# fig1 = plt.figure(1)
# plot_map_2d((Xc, Yc), logK[:,1], (nx, ny))

# K1 = mu + np.matmul(V, alpha[:,1])
# fig2 = plt.figure(2)




