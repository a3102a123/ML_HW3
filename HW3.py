import numpy as np
import time
import matplotlib.pyplot as plt
import sys

from sympy import false, true

is_random = False
is_test = False
dis_num = 3
seed = 521
threshold = 0.00001

N = 10000
np.set_printoptions(suppress=True)
np.set_printoptions(precision=10)

# Box–Muller method
# output a numpy array which size is n
def normal_generator(mu,sigma2,n = 1):
    sigma = np.sqrt(sigma2)
    U = np.random.uniform(size = n)
    V = np.random.uniform(size = n)
    fac = np.sqrt(-2 * np.log(U))
    Theta = 2 * np.pi * V
    X = fac * np.cos(Theta)
    Y = fac * np.sin(Theta)
    return sigma * X + mu 

def poly_generator(a,W):
    x = np.random.uniform(-1,1,1)
    # e
    y = normal_generator(0,a)[0]
    for i,w in enumerate(W):
        y += w*x**i
    return x, y

def Sequential(mu,sigma2):
    mean = 0
    var = 0
    sample = [1.220492527761238 ,  3.6967805272943366 , 2.7258100985704146 , 2.2138523069477527 , 2.2113035958584453 , 0.05399706095719692 , 4.3538771826058]
    for i in range(N):
        x = normal_generator(mu,sigma2)[0]
        if is_test:
            if i >= len(sample):
                break
            x = sample[i]
        n = i+1
        pre_mean = mean
        mean += (x - mean) / n
        var += ((x - pre_mean)*(x - mean) - var) / n
        print("Add data point: {}".format(x))
        print("Meat = {}Variance = {}".format(f'{mean:<20}',f'{var:<20}'))

def calc_poly(X,W,var = None):
    Y = []
    W_ori = W.copy()
    v = 0
    for x in X:
        if type(var) != type(None):
            v = calc_var(x,var)
            W = W_ori + v.reshape(4,1)
        y = 0
        for i,w in enumerate(W):
            y += x**i * w
        Y.append(y)
    return np.array(Y)

def calc_var(x,var):
    data = []
    for i in range(len(var)):
        data.append(x**i)
    data = np.array(data)
    return  var @ data

def draw_ground_truth(W,var):
    X = np.linspace(-2,2,num = 1000)
    Y = calc_poly(X,W)
    plt.title("Ground truth")
    plt.plot(X,Y,'k')
    plt.plot(X,Y + var,'r')
    plt.plot(X,Y - var,'r')

def Baysian(gb,ga,W):
    # test data
    sample_x = [-0.64152 , 0.07122 , -0.19330]
    sample_y = [0.19039 ,  1.63175 , 0.24507]
    sample_len = len(sample_x)

    # array for storing data
    data = []
    W10 = None
    W50 = None
    W10_var = None
    W50_var = None

    n = len(W)
    # the mean & var of W coefficient
    # w_co_var = the inverse of covariance
    I = np.identity(n)
    w_co_var = np.linalg.inv(gb*I) 
    w_co_var_inv = np.linalg.inv(gb*I)
    w_mu = np.zeros((n,1))
    pre_predict_y = 0
    # the variance factor of initial distribution
    a = gb
    # the design matrix X
    X = np.zeros((1,n))
    # the mean & var of Predictive distribution ~ N(mean , var)
    next_mean = 0.0
    mean = 0.0
    var = 0.0
    pre_var = 0.0
    is_conv = false
    for i in range(N):
        x , y = poly_generator(ga,W)
        if is_test:
            if i >= sample_len:
                break
            x = sample_x[i]
            y = sample_y[i]
        for k in range(n):
            X[0,k] = x**k
        b = w_co_var_inv
        S = w_co_var_inv
        # maybe b@I = b@w_co_var_inv in this case b should be a constan
        w_co_var_inv = a * (X.T @ X) + b@I
        w_co_var = np.linalg.inv(w_co_var_inv)
        w_mu = w_co_var @ (a * X.T * y + S@w_mu)
        # calc the predictive distribution of prior?
        # the mean & variance of predict Y?
        mean = next_mean
        predict_y = X @ w_mu
        next_mean += (predict_y - next_mean) / (i+1)
        next_mean = next_mean.item(0)
        pre_var = var
        var = 1.0 / a + X @ np.linalg.inv(S) @ X.T
        var = var.item(0)

        # check convergence
        diff =  abs(pre_var - var)
        if diff < threshold:
            is_conv = true

        if i < dis_num or i == (N - 1) or is_conv:
            print("Add data point ({}, {}):".format(x,y),"\n")
            print("X : \n",X,"\nPredict Y : ",X @ w_mu)

            print("Postirior mean:\n",w_mu,"\n")
            print("Posterior variance:\n",w_co_var,"\n")
            print("Predictive distribution ~ N({:.5f}, {:.5f})".format(mean,var) , "\n")
        # storing data
        data.append([x,y])
        if (i + 1) == 10:
            W10 = w_mu.copy()
            W10_var = w_co_var.copy()
        elif (i + 1) == 50:
            W50 = w_mu.copy()
            W50_var = w_co_var.copy()

        pre_predict_y = predict_y
        if is_conv:
            print("finish in {}".format(i))
            break
        # break
    if is_test:
        return
    
    # plot the result
    X = np.linspace(-2,2,num = 1000)
    plt.figure(figsize=(8, 6), dpi=120)
    plt.subplot(221)
    plt.axis((min(X),max(X),-18,22))
    draw_ground_truth(W,ga)

    plt.subplot(222)
    plt.axis((min(X),max(X),-18,22))
    data = np.array(data)
    Y = calc_poly(X,w_mu)
    data_x = data[:,0]
    data_y = data[:,1]
    Y_var = calc_poly(X,w_mu,w_co_var)
    Y_var_n = calc_poly(X,w_mu,-w_co_var)
    plt.title("Predict result")
    plt.plot(X,Y,'k')
    plt.plot(X,Y_var,'r')
    plt.plot(X,Y_var_n,'r')
    plt.scatter(data_x,data_y,color='b',s=15)

    plt.subplot(223)
    plt.axis((min(X),max(X),-18,22))
    Y = calc_poly(X,W10)
    data_10 = data[0:10]
    data_x = data_10[:,0]
    data_y = data_10[:,1]
    Y_var = calc_poly(X,W10,W10_var)
    Y_var_n = calc_poly(X,W10,-W10_var)
    plt.title("After 10 incomes")
    plt.plot(X,Y,'k')
    plt.plot(X,Y_var,'r')
    plt.plot(X,Y_var_n,'r')
    plt.scatter(data_x,data_y,color='b',s=15)

    plt.subplot(224)
    plt.axis((min(X),max(X),-18,22))
    Y = calc_poly(X,W50)
    data_50 = data[0:50]
    data_x = data_50[:,0]
    data_y = data_50[:,1]
    Y_var = calc_poly(X,W50,W50_var)
    Y_var_n = calc_poly(X,W50,-W50_var)
    plt.title("After 50 incomes")
    plt.plot(X,Y,'k')
    plt.plot(X,Y_var,'r')
    plt.plot(X,Y_var_n,'r')
    plt.scatter(data_x,data_y,color='b',s=15)
    plt.show()

# main
if is_random:
    seed = int(time.time())
np.random.seed(seed)
Baysian(1,1,[1,2,3,4])
sys.exit()
### Sequential Estimator
print("---Sequential Estimator---")
print("Input mu : ")
m = float(input())
print("Input Sequential Estimator variance : ")
s = float(input())
print("Data point source function: N({}, {})".format(m,s))
Sequential(m,s)

### Baysian Linear regression
print("---Input Baysian Linear regression---")
print("Inpur b : ")
b = int(input())
print("Input n : ")
n = int(input())
print("Input a (input single number one time): ")
a = []
for i in range(n):
    a.append(int(input()))
sys.exit()