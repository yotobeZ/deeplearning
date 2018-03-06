import numpy as np

#sigmoid function
def nonline(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#input dataset
X = np.array([ [0,0,1],
               [0,1,1],
               [1,0,1],
               [1,1,1] ])

#output dataset
y = np.array([[0,0,1,1]]).T

#send random numbers to make calculation
#deterministic(just a good practice)
np.random.seed(1)

#initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1))-1

for iter in range(10000):
    #forward propagation
    l0 = X
    l1 = nonline(np.dot(l0,syn0))

    #how much did we miss?
    l1_error = y-l1

    #multiply how much we missed by the
    #slope of the sigmoid at values in l1
    #为什么用斜率代表误差大小？
    l1_delta = l1_error*nonline(l1,True)

    #update weights (3*4)*(4*1)=3*1
    syn0 += np.dot(l0.T,l1_delta)
print("Output After Training:")
print(l1)










