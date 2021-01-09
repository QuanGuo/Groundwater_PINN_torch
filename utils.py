import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable, grad
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import rcParams
import os
from scipy.spatial import distance
from scipy.linalg import cholesky, eigh
import pandas as pd

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class HeadAlphaNet(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(HeadAlphaNet, self).__init__()

        self.fc1 = nn.Linear(n_feature, n_hidden[0])   # hidden layer 1
        self.fc2 = nn.Linear(n_hidden[0], n_hidden[1])   # hidden layer 2
        self.fc3 = nn.Linear(n_hidden[1], n_hidden[2])   # hidden layer
        self.fc4 = nn.Linear(n_hidden[2], n_hidden[3])   # hidden layer
        self.fc5 = nn.Linear(n_hidden[3], n_output)   # output layer

    def forward(self, x):

        x = torch.tanh(self.fc1(x))     # activation function for hidden layer
        x = torch.tanh(self.fc2(x))    # activation function for hidden layer
        x = torch.tanh(self.fc3(x))    # activation function for hidden layer
        x = torch.tanh(self.fc4(x))    # activation function for hidden layer
        x = self.fc5(x)    # activation function for hidden layer

        return x



def split_sample_in_order(X, y, k=0.8, dtype=torch.float32):
    '''
        Attention:
        X, y must be 2D np.array or torch.Tensor
    '''
    n = X.shape[0]
    train_num = int(np.round(n * k))
    if type(X) == torch.Tensor:
        if X.dtype != dtype:
            X = X.type(dtype)
            y = y.type(dtype)
    else: 
        X = torch.from_numpy(X).type(dtype)
        y = torch.from_numpy(y).type(dtype)

    train_X = X[:train_num, :]
    train_y = y[:train_num, :]
    valid_X = X[train_num:, :]
    valid_y = y[train_num:, :]

    return (train_X, train_y, valid_X, valid_y, X, y)

def flatten_mat(mat=None):
    mat_flat = mat.reshape(1, -1)
    return mat_flat

def recover_mat(mat_flat=None, shp=None):
    mat = mat_flat.reshape(shp)
    return mat

def add_coors_on_feature(X, coors):
    Xc, Yc = coors
    X = np.hstack((Xc, Yc, X))
    return X

def add_noise_on_realization(z, magnitude):
    wn = np.random.rand(z.shape[0], z.shape[1])*magnitude*2 + (1-magnitude)
    z = np.multiply(z, wn)
    return z

def distance_matrix(coor1, coor2=None):
    if len(coor1) > 1:
        coords1 = np.hstack(coor1)
    else:
        coords1 = np.reshape(coor1[0], (-1,1))
        print(coords1.shape)
    if coor2:
        if len(coor2) > 1:
            coords2 = np.hstack(coor2)
        else:
            coords2 = np.reshape(coor2[0], (-1,1))
    else:
        coords2 = coords1

    hmatrix = distance.cdist(coords1, coords2)
    return hmatrix

def chol_realization_generation(Q, mu=0, NR=1):
    B = cholesky(Q, lower=True)
    u = np.ones((Q.shape[1], NR))
    for i in range(NR):
        u[:,i] = np.random.normal(0, 1, Q.shape[1])
    y = mu + np.matmul(B, u)
    return y

def pca_realization_generation(Q, k=10, mu=0, NR=1):
    V, D = KEigDescend(Q,k)
    # V = np.matmul(V, np.diag(np.sqrt(D)))
    for i in range(k):
        V[:,i] = V[:,i]*np.sqrt(D[i])
    u = np.ones((k, NR))
    for i in range(NR):
        u[:,i] = np.random.normal(0, 1, k)
    y = mu + np.matmul(V, u)
    return y, u

def KEigDescend(M, k):
    D, V = eigh(M)
    for i in range(M.shape[0]):
        if D[i] < 0:
            D[i] = -D[i]
            V[:,i] = -V[:,i]
    idx = D.argsort()[::-1]   
    D = D[idx]
    V = V[:,idx]
    V = V[:,0:k]
    D = D[0:k]

    return V, D

def sample_location(m=2, LB=0, RB=1, Distribution="Even"):
    if Distribution == "Random":
        xrand = np.random.rand(m)*(RB-LB) + LB
        x = np.sort(xrand)
    elif Distribution == "Even":
        x = np.linspace(LB, RB, m, dtype=float)
    else:
        print("Unrecognized Distributuion")
        return None
    X = x.reshape(-1, 1)
    hmatrix = distance.cdist(X, X)
    return (x, hmatrix)

def sample_location_2d(nx=2, ny=2, xmin=0, xmax=1, ymin=0, ymax=1):
    stepx = float(xmax-xmin)/(nx-1)
    stepy = float(ymax-ymin)/(ny-1)
    [VX, VY] = np.mgrid[xmin:xmax+0.0001:stepx, ymin:ymax+0.0001:stepy]
    X = VX.reshape(-1, 1)
    Y = VY.reshape(-1, 1)
    return (X, Y)

def sample_idx_2d(nx=2, ny=2, xmin=0, xmax=1, ymin=0, ymax=1):
    Xi, Yi = sample_location_2d(nx, ny, xmin, xmax, ymin, ymax)
    Xi = np.floor(Xi).astype(int)
    Yi = np.floor(Yi).astype(int)
    return Xi, Yi
    
def plot_map_2d(x, y, z, shp=None):
    fig = plt.figure(figsize=(5.5,3.5), dpi=100)
    nx, ny = shp
    X,Y = np.meshgrid(x,y)
    z = recover_mat(z, (nx,ny))
    plt.pcolormesh(X,Y,z)
    plt.colorbar()
    return None

def compare_true_pred(y_true, y_pred, x, y, idx=None, z_bgd=None):

    f, axs = plt.subplots(1,2, figsize=(6,2.5))

    nx = len(x)
    ny = len(y)

    X,Y = np.meshgrid(x,y)

    if z_bgd is not None:
        z0 = z_bgd
    else:
        z0 = np.zeros((nx*ny,))-1

    ymin = np.amin(y_true)
    ymax = np.amax(y_true)

    if idx is None:
        z0 = y_true
    else:
        z0[idx] = y_true

    z = recover_mat(z0, (nx, ny))

    im1 = axs[0].pcolormesh(X, Y, z, vmin=ymin, vmax=ymax)
    axs[0].set_title('y true')

    if idx is None:
        z0 = y_pred
    else:
        z0[idx] = y_pred

    z = recover_mat(z0, (nx, ny))
    im2 = axs[1].pcolormesh(X,Y,z, vmin=ymin, vmax=ymax)
    axs[1].set_title('y pred')

    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(im1, cax=cbar_ax)

    return f

def plot_3D(x, y, p, figname):
    '''Creates 3D plot with appropriate limits and viewing angle

    Parameters:
    ----------
    x: array of float
        nodal coordinates in x
    y: array of float
        nodal coordinates in y
    p: 2D array of float
        calculated potential field

    '''
    fig = plt.figure(figsize=(5,3), dpi=100)
    ax = fig.gca(projection='3d')
    X,Y = np.meshgrid(x,y)
    surf = ax.plot_surface(X,Y,p[:], rstride=1, cstride=1, cmap=cm.viridis,
            linewidth=0, antialiased=False)

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.view_init(30,45)
    ax.set_title(figname)

def p_analytical(x, y):
    X, Y = np.meshgrid(x,y)

    p_an = np.sinh(1.5*np.pi*Y / x[-1]) /\
    (np.sinh(1.5*np.pi*y[-1]/x[-1]))*np.sin(1.5*np.pi*X/x[-1])

    return p_an

# def figsize(scale, nplots = 1):
#     fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
#     inches_per_pt = 1.0/72.27                       # Convert pt to inch
#     golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
#     fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
#     fig_height = nplots*fig_width*golden_mean              # height in inches
#     fig_size = [fig_width,fig_height]
#     return fig_size

# pgf_with_latex = {                      # setup matplotlib to use latex for output
#     "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
#     "text.usetex": True,                # use LaTeX to write all text
#     "font.family": "serif",
#     "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
#     "font.sans-serif": [],
#     "font.monospace": [],
#     "axes.labelsize": 10,               # LaTeX default is 10pt font.
#     "font.size": 10,
#     "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8,
#     "figure.figsize": figsize(1.0),     # default fig size of 0.9 textwidth
#     "pgf.preamble": [
#         r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
#         r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
#         ]
#     }
# mpl.rcParams.update(pgf_with_latex)

# def newfig(width, nplots = 1):
#     fig = plt.figure(figsize=figsize(width, nplots))
#     ax = fig.add_subplot(111)
#     return fig, ax

# def savefig(filename, crop = True):
#     if crop == True:
# #        plt.savefig('{}.pgf'.format(filename), bbox_inches='tight', pad_inches=0)
#         plt.savefig('{}.pdf'.format(filename), bbox_inches='tight', pad_inches=0)
#         plt.savefig('{}.eps'.format(filename), bbox_inches='tight', pad_inches=0)
#     else:
# #        plt.savefig('{}.pgf'.format(filename))
#         plt.savefig('{}.pdf'.format(filename))
#         plt.savefig('{}.eps'.format(filename))