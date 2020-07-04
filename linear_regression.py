import numpy as np
import matplotlib.pyplot as plt
class LinearRegression():

    def __init__(self,iterations=1500,alpha=0.01):
        self.iterations =iterations
        self.alpha = alpha
        # self.theta = np.array([2,1]).reshape(2,1)

    # need to train first
    def Train(self,training_set):

        self.data=np.loadtxt(fname=training_set,delimiter=',')
        shape=self.data.shape
        self.m=shape[0]
        print("Imported (Numpy converted) data size is: {} \nAnd datatype is: {}".format(shape,type(self.data)))
        print('Plotting Data ...\n')
        self.Ploting()
        print("Run Cost function and Compute J")


        y = np.mat(self.data[:, 1]).reshape(self.m, 1)
        # create 97x1 matrix from orginial data
        arrayx = np.mat(self.data[:, 0]).reshape(self.m, 1)
        # create 97x1 ones matrix to add in first colume to simulate X0=1
        ones = np.ones((self.m, 1))
        # use hstack to add ones to 1st colume and slice
        # from orginal data to 2nd colume
        X = np.hstack((ones, arrayx))

        theta = np.array([0,0]).reshape(2,1)
        J=self.ComputerCost(X,y,theta)
        print('with theta [0;0], J is ',J)

        theta = np.array([-1, 2]).reshape(2, 1)
        J = self.ComputerCost(X, y, theta)
        print('with theta [-1;2], J is ', J)

        thetaList=self.gradientDescent(X,y,theta,self.alpha,self.iterations)
        JList=[]
        for item in thetaList:
            J = self.ComputerCost(X, y, item)
            JList.append(J)
            print(J)
        J=min(JList)
        print('After GradientDescent J is ', J)
        self.plotLine(thetaList[0],thetaList[1])

    def Ploting(self):
        # this slice output numpy.ndarray, might use matrix next time
        x=self.data[:,0]
        y=self.data[:,1]

        plt.scatter(x,y)
        plt.xlabel('population size in 10,000s')
        plt.ylabel("profit in $10,000s")
        plt.show()

    def ComputerCost(self,X,y,theta):


        #cost function
        step_a=1/(2*self.m)
        #  S=sum((H-y).^2) in octave
        step_b=sum((X*theta-y).A**2)
        J= step_a*step_b
        return J

    def gradientDescent(self,X,y,theta,alpha,num_iters):
        m=self.m
        thetaList=[]
        for i in range(1,num_iters):
            error= (X*theta) - y
            theta=theta-(alpha/m) *(X.transpose())*error
            print("Theta = ",theta)
            self.ComputerCost(X,y,theta)
            thetaList.append(theta)

        return thetaList
    def plotLine(self,x,y):

        plt.plot(x, y)

        plt.show()
