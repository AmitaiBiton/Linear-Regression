import numpy as np
import matplotlib.pyplot as plt
import csv

reader = csv.reader(open("cancer_data.csv" , "rt") , delimiter = ",")
temp = list(reader)
result = np.array(temp).astype("float")

# -------------------------------q2 h(x)
def linearRegression(vector_theta, V_X):
    ##return np.sum(vector_theta*V_X)
    preY= np.sum(V_X*vector_theta)
    return preY


# -------------------------------q3 J(theta)
def calculateJ(m_x, v_y, vector_theta):
    J = 0
    for i in range(len(m_x)):
        J = J + np.power(linearRegression(m_x[i], vector_theta) - v_y[i], 2)
    return J / (2 * len(m_x))


# -----------------------q4 gradient descent

def sumJ(m_x, v_y, vector_theta):
    return (1/(len(m_x))) * np.matmul(m_x.T ,np.matmul(m_x ,vector_theta)-v_y)

#--------------------- q5  gradient descent :
def gradient_descent(m_x , v_y , vector_theta,alpha,b):
    i = 0
    list_x = []
    list_y = []
    while i<100:
        j_g = sumJ(m_x , v_y , vector_theta)# get gradient of J
        new_vector_theta = b*vector_theta - alpha * j_g # regressoin of theta
        list_y.append(calculateJ(m_x , v_y , vector_theta))# append to y J
        list_x.append(i)
        if np.sqrt(np.sum(new_vector_theta*vector_theta)) < 0.0001:# epsilon==0.0001
            break
        if np.sqrt(np.sum(sumJ(m_x , v_y , new_vector_theta) - sumJ(m_x , v_y , vector_theta))**2) < 0.0001: # J[theta](k+1) - J[theta](k)< epsilon
            break
        i +=1
        vector_theta = new_vector_theta
    if b == 1:#q7
        str1 = "alpha = "
    else:
        str1 = "MOMENTUM alpha = "
    str2 = str(alpha)
    str3 = str1 + str2
    plt.plot(list_x,list_y,label=str3)



#############----- q6 -------###################
def gradient_descent_MiniBatch(m_x , v_y , vector_theta, alpha):
    i = 0
    k = 0
    N = (len(m_x))/11 ## 3047/11 =277
    list_x = []
    list_y = []
    while i<100:
        MiniBatch_m_x = m_x[int(k*N):int((k+1)*N),:]## T group -> N = M/T
        MiniBatch_v_y = v_y[int(k*N):int((k+1)*N)]
        j_g = sumJ(MiniBatch_m_x , MiniBatch_v_y , vector_theta)
        new_vector_theta = vector_theta - alpha * j_g
        list_y.append(calculateJ(m_x , v_y , vector_theta))
        list_x.append(i)
        if np.sqrt(np.sum(new_vector_theta*vector_theta)) < 0.0001:
            break
        if np.sqrt(np.sum(sumJ(m_x , v_y , new_vector_theta) - sumJ(m_x , v_y , vector_theta))**2) < 0.0001:
            break

        i +=1
        vector_theta = new_vector_theta
        if (k+1)*N != len(m_x):
            k+=1
        else:
            k=0
    str1 = "MiniBatch - alpha = "
    str2 = str(alpha)
    str3 = str1 + str2
    plt.plot(list_x,list_y,label=str3)




#--------------------Normalize --q1------------------------#
normResult = result.T
y = result[:,-1]
x = np.delete(result, -1,axis =1)
for i in range(len(normResult)):
    avg = np.average(normResult[i])
    std = np.std(normResult[i])
    normResult[i] = (normResult[i]-avg)/std

#------------------------Check avg and std--------------------------#

for i in normResult:
  print("avg")
  print(np.average(i))
  print("std")
  print(np.std(i))

############################end q1
for i in range(3):
    normResult = normResult.T
normY = normResult[:,-1]
normX = np.delete(normResult, -1,axis =1)
ones = np.ones((len(normX),1)) ##add another raw of ones
normX = np.hstack((ones,normX)) ##add another raw of one
vector_theta = np.ones(len(normX[0]))

#_____________________MAIN ____________________________


gradient_descent(normX , normY , vector_theta,0.1,1)
gradient_descent(normX , normY , vector_theta,0.01,1)
gradient_descent(normX , normY , vector_theta,0.001,1)
#---------------to check q6 need to run with three rows below
#gradient_descent_MiniBatch(normX , normY , vector_theta, 0.1)
#gradient_descent_MiniBatch(normX , normY , vector_theta, 0.01)
#gradient_descent_MiniBatch(normX , normY , vector_theta, 0.001)

## --------- to check q7 run with first three rows and three below
gradient_descent(normX , normY , vector_theta,0.1,0.9)
gradient_descent(normX , normY , vector_theta,0.01,0.9)
gradient_descent(normX , normY , vector_theta,0.001,0.9)


##-------------------------- display the Graph__________________
plt.xlabel('Iteration')
plt.ylabel('J(theta)')
plt.title('Graph')
plt.legend()
plt.show()