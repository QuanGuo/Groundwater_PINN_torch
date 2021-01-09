import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable, grad
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyDOE import lhs
from utils import *
import time

class PhysicsInformedNN(nn.Module):

    def __init__(self, layers):
        super(PhysicsInformedNN, self).__init__()
    
        self.layers = layers

        self.weights, self.biases = self.initialize_NN(layers)

        self.u_pred, self.f_pred = None, None

        self.loss = None

        self.loss_list = []       
        
        self.optimizer = torch.optim.LBFGS(params=self.weights+self.biases,
                                            lr=0.00001, max_iter=5000, max_eval=5000,
                                         tolerance_grad=1e-07, tolerance_change=1e-08,
                                          history_size=100, line_search_fn=None)


    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = Variable(torch.zeros([1,layers[l+1]], dtype=torch.float32), requires_grad=True)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):

        return Variable(nn.init.xavier_normal_(torch.empty(size[0], size[1])), requires_grad=True)
    
    def neural_net(self, x, y, weights, biases):

        num_layers = len(weights) + 1
        H = torch.cat((x,y),1)
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = torch.tanh(torch.add(torch.matmul(H, W), b))

        W = weights[-1]
        b = biases[-1]
        Y = torch.add(torch.matmul(H, W), b).requires_grad_()

        return Y

    def net_u(self, x, y):
        u = self.neural_net(x, y, self.weights, self.biases)
        return u
    
    def net_f(self, x, y):

        u = self.neural_net(x, y, self.weights, self.biases)

        u_x = grad(u.sum(), [x], create_graph=True)[0]
        u_y = grad(u.sum(), [y], create_graph=True)[0]

        u_yy = grad(u_y.sum(), y, create_graph=True)[0]

        u_xx = grad(u_x.sum(), x, create_graph=True)[0]

        f = u_yy + u_xx

        return f.requires_grad_(True)

    def forward(self, x_u, y_u, x_f, y_f):

        u_pred = self.net_u(x_u, y_u) 
        f_pred = self.net_f(x_f, y_f)   

        return u_pred, f_pred

    def predict(self, X_input):
        x_tensor = torch.tensor(X_input[:,0:1], dtype=torch.float32)
        y_tensor = torch.tensor(X_input[:,1:2], dtype=torch.float32)
        return self.neural_net(x_tensor,y_tensor, self.weights, self.biases).detach().numpy().squeeze()

    def loss_func(self, pred_tuple, true_tuple):
        u_pred, f_pred = pred_tuple
        u, f = true_tuple
        res_u = u - u_pred
        res_f = f - f_pred
        loss = torch.mean(res_u.pow(2))+torch.mean(res_f.pow(2))
        return loss.requires_grad_()
    
    def customized_backward(self, loss, params):
        grads = grad(loss, params, retain_graph=True)
        for vid in range(len(params)):
            params[vid].grad = grads[vid]
        return grads

    def callback(self, loss):
        self.loss_list.append(loss)

    def train_LBFGS(self, u_data, f_data, loss_func, optimizer):
        x_u_tensor, y_u_tensor, u_tensor = u_data
        x_f_tensor, y_f_tensor, f_tensor = f_data
        def closure():

            optimizer.zero_grad()
            u_pred, f_pred = self.forward(x_u_tensor, y_u_tensor, x_f_tensor, y_f_tensor)
            loss = loss_func((u_pred, f_pred), (u_tensor, f_tensor)) #.requires_grad_()
            
            self.callback(loss)
            if np.remainder(len(self.loss_list),100) == 0:
                print('Iter #', len(self.loss_list), 'Loss:', self.loss_list[-1].detach().numpy().squeeze())
            
            g = self.customized_backward(loss, self.weights+self.biases)
            # loss.backward(retain_graph=True)

            return loss

        optimizer.step(closure)
        self.u_pred, self.f_pred = self.forward(x_u_tensor, y_u_tensor, x_f_tensor, y_f_tensor)

        self.loss = loss_func((self.u_pred, self.f_pred), (u_tensor, f_tensor))

    def train(self, epoch, u_data, f_data, loss_func, optimizer):
        x_u_tensor, y_u_tensor, u_tensor = u_data
        x_f_tensor, y_f_tensor, f_tensor = f_data
        for i in range(epoch):
            optimizer.zero_grad()
            u_pred, f_pred = self.forward(x_u_tensor, y_u_tensor, x_f_tensor, y_f_tensor)
            loss = loss_func((u_pred, f_pred), (u_tensor, f_tensor)) #.requires_grad_()
            
            self.callback(loss)
            if np.remainder(len(self.loss_list),100) == 0:
                print('Iter #', len(self.loss_list), 'Loss:', self.loss_list[-1].detach().numpy().squeeze())
            
            g = self.customized_backward(loss, self.weights+self.biases)
            # loss.backward()

  
            optimizer.step()

        self.u_pred, self.f_pred = self.forward(x_u_tensor, y_u_tensor, x_f_tensor, y_f_tensor)

        self.loss = loss_func((self.u_pred, self.f_pred), (u_tensor, f_tensor))

    def coor_shift(self, X, lbs, ubs):

        return 2.0*(X - lbs) / (ubs - lbs) - 1

    def data_loader(self, X, u, lbs, ubs):
                
        X = self.coor_shift(X, lbs, ubs)

        x_tensor = torch.tensor(X[:,0:1], requires_grad=True, dtype=torch.float32)
        y_tensor = torch.tensor(X[:,1:2], requires_grad=True, dtype=torch.float32)

        u_tensor = torch.tensor(u, dtype=torch.float32)

        return (x_tensor, y_tensor, u_tensor)

if __name__ == "__main__": 
       

    N_u = 100
    N_f = 20*20
    N_uc = 30

    nx = 61
    ny = 61

    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny)

    Exact = p_analytical(x,y)

    X, Y = np.meshgrid(x,y)
    
    X_star = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              

    # Domain bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    
    lbs = np.array([0,0])
    ubs = np.array([1,1])
    # top
    xx1 = np.hstack((X[0:1,:].T, Y[0:1,:].T))
    uu1 = Exact[0:1,:].T

    # left
    xx2 = np.hstack((X[:,0:1], Y[:,0:1]))
    uu2 = Exact[:,0:1]

    # bottom
    xx3 = np.hstack((X[-1:,:].T, Y[-1:,:].T))
    uu3 = Exact[-1:,:].T

    # right
    xx4 = np.hstack((X[:,-1:], Y[:,-1:]))
    uu4 = np.zeros((xx4.shape[0],1))

    X_u_train = np.vstack([xx1, xx2, xx3])
    u_train = np.vstack([uu1, uu2, uu3])

    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    f_train = np.zeros((X_f_train.shape[0],1))
    
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx, :]

    # idx2 = np.random.choice(xx4.shape[0], N_uc, replace=False)
    # X_bc_train = xx4[idx2,:]
    # ux_bc_train = uu4[idx2,:]

    BCs = [1]

    layers = [2, 20, 20, 20, 20, 20, 20, 1]

    model = PhysicsInformedNN(layers)
    u_data = model.data_loader(X_u_train, u_train, lbs, ubs)
    f_data = model.data_loader(X_f_train, f_train, lbs, ubs)

    start_time = time.time() 
    optimizer = torch.optim.LBFGS(params=model.weights+model.biases,
                                    lr=0.001, max_iter=3000, #max_eval=4000,
                                    tolerance_grad=1e-05, tolerance_change=1e-07,
                                    history_size=10, line_search_fn=None)

    model.train_LBFGS(u_data, f_data, model.loss_func, optimizer)

    # optimizer = torch.optim.Adam(model.weights+model.biases, lr=1e-5)
    # model.train(1000, u_data, f_data, model.loss_func, optimizer)

    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))

    X_pred = model.coor_shift(X_star, lbs, ubs)
    u_pred = model.predict(X_pred)
            
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    print('Error u: %e' % (error_u))                     

    np.savetxt('u_pred_torch.txt', u_pred)