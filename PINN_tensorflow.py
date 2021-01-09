import numpy as np
import matplotlib.pyplot as plt
from utils import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import rcParams

import tensorflow as tf
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
# from plotting import newfig, savefig
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(1234)
tf.set_random_seed(1234)

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16



class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_u, u, X_f, X_c, u_c, layers, lb, ub, nu):
        
        self.lb = lb
        self.ub = ub
    
        self.x_u = X_u[:,0:1]
        self.t_u = X_u[:,1:2]
        
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        
        self.u = u
        
        self.layers = layers
        self.nu = nu
        
        self.x_c = X_c[:,0:1]
        self.t_c = X_c[:,1:2]

        self.u_c = u_c
        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])        
                
        self.x_c_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.t_c_tf = tf.placeholder(tf.float32, shape=[None, self.t_c.shape[1]])        
        self.u_c_tf = tf.placeholder(tf.float32, shape=[None, self.u_c.shape[1]])


        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf) 
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)         
        self.uc_pred = self.net_uc(self.x_c_tf, self.t_c_tf)  

        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred)) + \
                    tf.reduce_mean(tf.square(self.u_c - self.uc_pred))
               
                
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 5000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

                
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
            
    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        return u
    
    def net_f(self, x, t):
        u = self.net_u(x,t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_tt = tf.gradients(u_t, t)[0]
        u_xx = tf.gradients(u_x, x)[0]
        # f = u_t + u*u_x - self.nu*u_xx
        f = u_tt + u_xx
        return f

    def net_uc(self, x, t):
        u = self.net_u(x,t)
        u_x = tf.gradients(u, x)[0]
        return u_x

    def callback(self, loss):
        L = []
        L.append(loss)
        # print('Loss:', loss)

        
    def train(self):
        
        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f, self.x_c_tf: self.x_c,
                   self.t_c_tf: self.t_c, self.u_c_tf: self.u_c }


        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)        
                                    
    
    def predict(self, X_star):
                
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:,0:1], self.t_u_tf: X_star[:,1:2]})  
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]})
        uc_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]})
        return u_star, f_star, uc_star
    
if __name__ == "__main__": 
     
    nu = 0.01/np.pi
    noise = 0.0        

    N_u = 100
    N_f = 30*30
    N_uc = 30
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    
    # data = scipy.io.loadmat('../Data/burgers_shock.mat')
    # t = data['t'].flatten()[:,None]
    # x = data['x'].flatten()[:,None]
    # Exact = np.real(data['usol']).T

    nx = 61
    ny = 61

    x = np.linspace(0,1,nx)
    t = np.linspace(0,1,ny)

    Exact = p_analytical(x,t)

    X, T = np.meshgrid(x,t)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    

    # top
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
    uu1 = Exact[0:1,:].T

    # left
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))
    uu2 = Exact[:,0:1]

    # bottom
    xx3 = np.hstack((X[-1:,:].T, T[-1:,:].T))
    uu3 = Exact[-1:,:].T
    print(uu3.shape)

    # right
    xx4 = np.hstack((X[:,-1:], T[:,-1:]))
    # uu4  = np.sin(1.5*np.pi*X[:,-1:] / x[-1]).reshape((xx4.shape[0],1))
    uu4 = np.zeros((xx4.shape[0],1))
    print(uu4.shape)

    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])
    
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]
        
    idx2 = np.random.choice(xx4.shape[0], N_uc, replace=False)
    X_bc_train = xx4[idx2,:]
    ux_bc_train = uu4[idx2,:]

    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, X_bc_train, ux_bc_train, layers, lb, ub, nu)
    
    start_time = time.time()                
    model.train()
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
    u_pred, f_pred, uc_pred = model.predict(X_star)
            
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    print('Error u: %e' % (error_u))                     

    np.savetxt('u_pred.txt', u_pred)
    # U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    # Error = np.abs(Exact - U_pred)

    # plot_3D(x,t,Exact, './Exact.png')
    # plot_3D(x,t,u_pred, './u_pred.png')
    # plot_map_2d((X,T), Error, (nx, ny))
    # plt.show()


