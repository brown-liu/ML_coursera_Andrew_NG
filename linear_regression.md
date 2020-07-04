<h2>Linear regression</h2>
linear regression is a very basic but useful algorithem which can learn from the
training set and predict the outcome for the test set.

<h3>Hepothesis Function</h3>
Linear regression is used under the assumption that the hepothesis for certain event is linear, and will look
like something like this:

y=Hepothesis(x)=  theta_1*X_1  + theta_2*X_2  +theta_3*X_3  ... theta_n* X_n

In above function, theta_n is the constant,  X_n is the input value of each feature. 

And above function can also be expressed as:

y=  Hepothesis(X) or H(X) = theta.T * X

now, the theta is the matrix of [theta_1
                                 theta_2 
                                 theta_3 
                                 .
                                 .
                                 .
                                 theta_n  ]
theta.T or theta.transpose() = [theta_1 theta_2 theta_3 ... theta_n ]
X is the matrix of [ X1
                     X2
                     X3
                     .
                     .
                     .Xn]    
So, y = theta_1*X_1  + theta_2*X_2  +theta_3*X_3  ... theta_n* X_n
can be rewritten as y= theta.T * X          

<h5>*Hepothesis or H(x) = theta.T*X where theta is the array of theta_1 to theta_n *  </h5>>             

<h3>Cost Function<h3/>  
Cost function, AKA "loss function","Squared error function" or "Mean squared error", is used to calculate the difference between the result
obtained from our Hepothesis function and the real data(training set). 
we will use J to represent the result from cost function.
We will get a better set of theta by minimizing J.
Without considering overfitting issue, the smaller the J, the better sets of thetas, hence a more ideal hepothesis function.

m = number of training_set

J(theta) = 1/(2*m) * sum ((y(hepothesis) - y(training_set))**2) 
         = 1/(2*m) * sum ((H(X) - y(training_set))**2)

<h3>Gradient Descent</h3>
We need to reduce the J so our hepothesis can fit well in the data.
Gradient Descent is one of the algorithems to reduce the J quickly.

alpha will be the parameter to adjust the step length.
i might be the 1st, 2nd, 3rd ...n_th feature in the theta set.

theta_1=theta_1 - alpha * derivative(J(theta1))
theta_2=theta_2 - alpha * derivative(J(theta2))
.......

<strong>warning: numpy dont seems to calculate derivative, so might need to use other lib:</strong>
************************************* EXAMPLE*************************************************
In [1]: from sympy import *
In [2]: import numpy as np
In [3]: x = Symbol('x')
In [4]: y = x**2 + 1
In [5]: yprime = y.diff(x)
In [6]: yprime
Out[6]: 2⋅x

In [7]: f = lambdify(x, yprime, 'numpy')
In [8]: f(np.ones(5))
Out[8]: [ 2.  2.  2.  2.  2.]
**********************************************************************************************
because derivative(J(theta_j)) = (H(X)-y)*Xj, function above can be re-written as:

theta_j=theta_j - (alpha /m) * sum(H(X)-y) * X_j

theta_j is the value in that theta matrix. it can have many dimensions.
alpha is the step length where we can always adjust.
m is the number of dataset or length.
X_j is the j_th(colume view) value in the row of the line of data we working on.  

when calculation, we need to record each theta_j and append to a list.

<h5> we now have a list of theta_j, and we can get various J from those theta_J</h5>

At the beginning, we said we want the smallest J. Now we can have it.
With the smallest J, we can also check what is the theta set behind it.
With the theta set in hand, we finally have the hepothsis function.

<h4> Notice</h4>
1. good to use matpoltlib to plot the data, so we can have a intuition what kind of data we dealing with.
2. Numpy(as np) has np.array and np.mat, be careful about the calculation.

