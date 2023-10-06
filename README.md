# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary library.

2.Load the text file in the compiler.

3.Plot the graphs using sigmoid , costfunction and gradient descent.

4.Predict the values.

5.End the Program.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Vijayaraj V
RegisterNumber: 212222230174 
*/


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data1 (2).txt",delimiter=',')
x=data[:,[0,1]]
y=data[:,2]

x[:5]

y[:5]

plt.figure()
plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted")
plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not admitted")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()
plt.show()

def signoid(z):
  return 1/(1+np.exp(-z))
  
plt.plot()
x_plot=np.linspace(-10,10,100)
plt.plot(x_plot,signoid(x_plot))
plt.show()

def costFunction(theta,x,y):
  h=signoid(np.dot(x,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return J,grad
  
x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
J,grad=costFunction(theta,x_train,y)
print(J)
print(grad)

x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,x_train,y)
print(J)
print(grad)

def cost(theta,x,y):
  h=signoid(np.dot(x,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return J

def gradient(theta,x,y):
  h=signoid(np.dot(x,theta))
  grad=np.dot(x.T,h-y)/x.shape[0]
  return grad

x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method="Newton-CG",jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,x,y):
  x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot=np.c_[xx.ravel(),yy.ravel()]
  x_plot=np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot=np.dot(x_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted")
  plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 Score")
  plt.ylabel("Exam 2 Score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,x,y)

prob=signoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,x):
  x_train=np.hstack((np.ones((x.shape[0],1)),x))
  prob=signoid(np.dot(x_train,theta))
  return (prob>=0.5).astype(int)
  
np.mean(predict(res.x,x)==y)


```

## Output:
![image](https://github.com/vijayarajv1704/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121303741/5f3a94ca-38c0-431e-a300-ed7bdeea5b6b)

![image](https://github.com/vijayarajv1704/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121303741/0ca69411-bc2e-4502-9f6f-9020f46062d1)

![image](https://github.com/vijayarajv1704/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121303741/41f6d8d8-efb0-4638-8b28-8a695885a292)

![image](https://github.com/vijayarajv1704/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121303741/b9c3d170-b2ec-4089-9f77-900010a754a5)

![image](https://github.com/vijayarajv1704/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121303741/81157976-1937-46eb-ba3b-21a5582b832d)

![image](https://github.com/vijayarajv1704/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121303741/b7345d09-233f-47cb-a55d-bbfa5091f984)

![image](https://github.com/vijayarajv1704/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121303741/da440d53-3256-4bae-8857-e45e39b705c7)

![image](https://github.com/vijayarajv1704/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121303741/71a2896f-189f-4ae3-bfe9-7c61dffd7983)

![image](https://github.com/vijayarajv1704/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121303741/8bc0f051-62ba-4623-b6f2-6b07724d641e)

![image](https://github.com/vijayarajv1704/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121303741/b9310915-44d8-42df-a9a8-2224a7384b7d)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

