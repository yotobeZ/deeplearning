import numpy as np

#sigmoid function
def nonline(x,deriv=False):
    if(deriv==True):
       return x*(1-x)
    return 1/(1+np.exp(-x))

def nonline1(x,deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return  1-(1 / (1 + np.exp(-x)))

#怎么用relu来拟合？relu在这里需要知道矩阵中每个元素的值判断和0的大小关系之后才可以决定在函数上的值，无法直接像sigmoid函数进行矩阵运算
#def nonline2(x,deriv=False):
    if  x<=0 :
        return 0
    if (deriv == True & x>0):
        return 1
    else:
     return  x


#input dataset
X = np.array([ [0,0,1],
               [0,1,1],
               [1,0,1],
               [1,1,1] ])

#output dataset
y = np.array([[0,0,1,1]]).T

#send random numbers to make calculation
#deterministic(just a good practice)
#此行定义后，之后每次生成的随机数都一样，seed里的数字变了，生成的随机数也改变
#最后的结果也改变了，seed不变则以上不会变
np.random.seed(1)

#initialize weights randomly with mean 0
#初始生成如下
# [[-0.16595599]
# [ 0.44064899]
# [-0.99977125]]
#2*和-1是干嘛的
syn0 = 2*np.random.random((3,1))-1
#print("syn0 : ")
#print(syn0)
#i = 1
for iter in range(10000):
    # print("i : "+str(i))
    #i=i+1
    #forward propagation
    #sigmoid在这里是激活函数？选用哪个函数来预测根据什么？选用sigmoid有什么科学性？
   #激活函数的选择也很重要，如果选择不合适的，如本例尝试sigmoid函数关于y轴对称的图形来做激活函数，
    #在10000次训练内效果极差，毫无拟合效果可言,即使增加到1000000次也没有拟合趋势，可见激活函数选择很重要
    l0 = X
    l1 = nonline(np.dot(l0,syn0))
   # print("l1 : "+str(l1))

    #how much did we miss?
    l1_error = y-l1
    # print("l1_error : " + str(l1_error))

    #multiply how much we missed by the
    #slope of the sigmoid at values in l1
    #为什么用斜率代表误差大小？绝对值越大或者越小，斜率就越小，代表误差越小，为什么？
    z=nonline(l1,True)
    # print("nonline(l1,True) : ")
    #print(nonline(l1,True))
    l1_delta = l1_error*z
    #print("l1_delta : " + str(l1_delta))

    #update weights (3*4)*(4*1)=3*1
    syn0 += np.dot(l0.T,l1_delta)
    # print("syn0 : " + str(syn0))
   # if(i%10000==0):
      #  print(str(i/10000)+" : ")
       # print(l1)

print("Output After Training:")
print(l1)


#看每一步的权值输出







