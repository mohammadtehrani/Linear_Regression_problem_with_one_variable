import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def generate_data():
    data = pd.read_csv('dataset1.txt')
    X = data['X'].to_numpy()
    Y = data['Y'].to_numpy()
    return X,Y

X,Y = generate_data()
print(f'X:{X}')
print(f'Y:{Y}')

X,Y = generate_data()
plt.scatter(X,Y)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

def close_form():
    X,Y = generate_data()
    n = len(X)
    
    #parameter Model
    slope_close = 0
    intercept_close = 0
    learning_rate_close = 0.01
    epochs_close = 1500
    
    #MSE
    for _ in range(epochs_close):
        Y_pred = slope_close * X + intercept_close
        mse = np.mean((Y_pred - Y) ** 2)
        gradient_slope_close = (2/n) * np.sum((Y_pred - Y) * X)
        gradient_intercept_close = (2/n) * np.sum(Y_pred - Y)
        
        #update parameter
        slope_close -= learning_rate_close * gradient_slope_close
        intercept_close -= learning_rate_close * gradient_intercept_close
        
    return slope_close,intercept_close

slope_close,intercept_close = close_form()
print(f'theta1 = {slope_close}')
print(f'theta0 = {intercept_close}')

slope_close,intercept_close = close_form()
p1 = slope_close * 6.2 + intercept_close
p2 = slope_close * 12.8 + intercept_close
p3 = slope_close * 22.1 + intercept_close
p4 = slope_close * 30 + intercept_close

print(f'p1 = {p1}')
print(f'p2 = {p2}')
print(f'p3 = {p3}')
print(f'p4 = {p4}')

slope_close,intercept_close = close_form()
plt.scatter(X,Y,label = 'data')
plt.plot(X,slope_close * X + intercept_close,color = 'red',label = 'Linear regression(close-form)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

def online_mode():
    X,Y = generate_data()
    n = len(X)
    
    #parameter Model
    slope_online = 0
    intercept_online = 0
    learning_rate_online = 0.01
    interations_online = 1500
    
    #Model training
    for _ in range(interations_online):
        random_index = np.random.randint(0,n)
        X_random = X[random_index]
        Y_random = Y[random_index]
        
        Y_pred = slope_online * X_random + intercept_online
        error = Y_pred - Y_random
        
        slope_online -= learning_rate_online * error * X_random
        intercept_online -= learning_rate_online * error
        
    return slope_online,intercept_online

slope_online,intercept_online = online_mode()
print(f'theta1 = {slope_online}')
print(f'theta0 = {intercept_online}')

slope_online,intercept_online = online_mode()
p1 = slope_online * 6.2 + intercept_online
p2 = slope_online * 12.8 + intercept_online
p3 = slope_online * 22.1 + intercept_online
p4 = slope_online * 30 + intercept_online

print(f'p1 = {p1}')
print(f'p2 = {p2}')
print(f'p3 = {p3}')
print(f'p4 = {p4}')


def predict(X,slope_online):
    h_theta = X*slope_online
    
    return h_theta

def cost_online(h_theta,Y):
    m=Y.shape[0]
    j = (1/(2*m))*np.dot((h_theta-Y).T,(h_theta-Y))
    return j

def grad(X,h_theta,Y):
    m=Y.shape[0]
    grad = (1/m)*((h_theta-Y).T*X).T
    return grad

learning_rate=0.01
iterations = 1500
j_history=[]

for i in range(iterations):
    m = Y.shape[0]
    h_theta = predict(X,slope_online)
    cost = cost_online(h_theta,Y)
    grad = (1/m)*((h_theta-Y).T*X).T
    slope_online = slope_online-learning_rate*grad
    j_history.append(cost)
    #print("Epoch :",i,"Cost :",cost)

x=np.linspace(0,iterations,iterations)
plt.ylabel('cost function') 
plt.plot(x,j_history,color='r') 
plt.xlabel('No. of iterations')
plt.title('Cost function VS iterations')

slope_online,intercept_online = online_mode()
plt.scatter(X,Y,label = 'data')
plt.plot(X,slope_online * X + intercept_online,color = 'red',label = 'Linear regression(online-mode)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


def batch_mode():
    X,Y = generate_data()
    n = len(X)
    
    #parameter Model
    slope_batch = 0
    intercept_batch = 0
    learning_rate_batch = 0.01
    iterations_batch = 1500
    
    #Model training
    for _ in range(iterations_batch):
        Y_pred = slope_batch * X + intercept_batch
        error = Y_pred - Y
        
        gradient_slope_batch = (2/n) * np.sum(error * X)
        gradient_intercept_batch = (2/n) * np.sum(error)
        
        slope_batch -= learning_rate_batch * gradient_slope_batch
        intercept_batch -= learning_rate_batch * gradient_intercept_batch
        
    return slope_batch,intercept_batch

slope_batch,intercept_batch = batch_mode()
p1 = slope_batch * 6.2 + intercept_batch
p2 = slope_batch * 12.8 + intercept_batch
p3 = slope_batch * 22.1 + intercept_batch
p4 = slope_batch * 30 + intercept_batch

print(f'p1 = {p1}')
print(f'p2 = {p2}')
print(f'p3 = {p3}')
print(f'p4 = {p4}')

def predict(X,slope_batch):
    h_theta = X*slope_batch
    
    return h_theta

def cost_batch(h_theta,Y):
    m=Y.shape[0]
    j = (1/(2*m))*np.dot((h_theta-Y).T,(h_theta-Y))
    return j

def grad(X,h_theta,Y):
    m=Y.shape[0]
    grad = (1/m)*((h_theta-Y).T*X).T
    return grad

learning_rate=0.01
iterations = 1500
j_history=[]

for i in range(iterations):
    h_theta = predict(X,slope_batch)
    cost = cost_batch(h_theta,Y)
    grad = (1/m)*((h_theta-Y).T*X).T
    slope_batch = slope_batch-learning_rate*grad
    j_history.append(cost)
    #print("Epoch :",i,"Cost :",cost)

x=np.linspace(0,iterations,iterations)
plt.ylabel('cost function') 
plt.plot(x,j_history,color='r') 
plt.xlabel('No. of iterations')
plt.title('cost function VS iterations')

slope_batch,intercept_batch = batch_mode()
plt.scatter(X,Y,label = 'data')
plt.plot(X,slope_batch * X + intercept_batch,color = 'red',label = 'Linear regression(batch-mode)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

slope_close,intercept_close = close_form()
slope_online,intercept_online = online_mode()
slope_batch,intercept_batch = batch_mode()
plt.scatter(X,Y,label = 'data')
plt.plot(X,slope_online * X + intercept_online,color = 'red',label = 'Linear regression(online-mode)')
plt.plot(X,slope_batch * X + intercept_batch,color = 'green',label = 'Linear regression(batch-mode)')
plt.plot(X,slope_close * X + intercept_close,color = 'grey',label = 'close-form',linestyle='dotted')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

