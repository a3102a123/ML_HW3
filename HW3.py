import numpy as np
import time
import matplotlib.pyplot as plt
import sys

is_random = False
seed = 521

# Box–Muller method
def normal_generator(mu,sigma,n = 1):
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
    y = normal_generator(0,a) 
    for w in W:
        y += x*w
    return y

# main
if is_random:
    seed = int(time.time())
np.random.seed(seed)
### Sequential Estimator
print("---Sequential Estimator---")
print("Input mu : ")
m = int(input())
print("Input Sequential Estimator sigma : ")
s = int(input())
print("Data point source function: N({}, {})".format(m,s))

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