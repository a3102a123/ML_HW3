import numpy as np
import time
import matplotlib.pyplot as plt
import sys

is_random = False
is_test = True
seed = 521

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
    return y

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

def Baysian(gb,ga,W):
    # test data
    sample_x = [-0.64152 , 0.07122 , -0.19330]
    sample_y = [0.19039 ,  1.63175 , 0.24507]
    sample_len = len(sample_x)

    n = len(W)
    # the mean & var of W coefficient
    # w_co_var = the inverse of covariance
    I = np.identity(n)
    w_co_var = np.linalg.inv(gb * I)
    w_co_var_inv = np.linalg.inv(gb * I)
    w_mu = np.zeros((n,1))
    # the design matrix X
    X = np.zeros((1,n))
    # the mean & var of Predictive distribution ~ N(mean , var)
    mean = 0
    var = 0
    for i in range(N):
        if is_test:
            if i >= sample_len:
                break
            x = sample_x[i]
            y = sample_y[i]
        for k in range(n):
            X[0,k] = x**k
        print("Add data point ({}, {}):".format(x,y),"\n")
        print("mean / var : ",mean, " / " , var )
        if var == 0:
            var = 1
        a = 1 / var
        b = w_co_var_inv
        S = w_co_var_inv
        print(X)
        w_co_var_inv = a * (X.T @ X) + b@I
        w_co_var = np.linalg.inv(w_co_var_inv)
        w_mu = w_co_var @ (a * X.T * y + S@w_mu)
        print("Predict Y : ",X @ w_mu)
        mean = 0.00000
        var = 1.0
        print("Postirior mean:\n",w_mu,"\n")
        print("Posterior variance:\n",w_co_var,"\n")
        print("Predictive distribution ~ N({}, {})".format(mean,var) , "\n")

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