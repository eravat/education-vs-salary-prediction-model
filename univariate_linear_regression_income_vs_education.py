import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients
import numpy as np


data = pd.read_csv("Earnings20Education20Sets.csv")
x_train = data["education"].values          #years of education
y_train = data["yearly_salary"].values      #annual salary assuming 40 hours/week for 50 weeks/year, calculated from hourly earnings adjusted for tax and such                           #m == 2950 or in other words, units of data is 2950

def calculate_cost(x,y,w,b):
    m = x.shape[0]
    total_cost = 0
    for i in range(m):
        squared_error = (w*x[i] + b - y[i])**2
        total_cost += squared_error
    cost = total_cost/(2*m)
    return cost

def calculate_slope(x,y,w,b):
    m = x.shape[0]
    dj_dw = dj_db = 0

    for i in range(m):
        dj_dw_i = (w*x[i] + b - y[i]) * x[i]        #partial derivative of cost function wrt. w for x^(i)
        dj_db_i = w*x[i] + b - y[i]                 #partial derivative of cost function wrt. b for x^(i)
        dj_dw += dj_dw_i                            #summing up 
        dj_db += dj_db_i                            
    dj_dw = dj_dw/m                                 #averaging
    dj_db = dj_db/m
    return dj_dw, dj_db



def gradient_descent(x,y,w_in,b_in,alpha,number_of_iterations):
    J_history = []      #for graphing
    p_history = []      #for graphing
    final_w = w_in
    final_b = b_in
    for i in range(number_of_iterations):
        dj_dw, dj_db = calculate_slope(x,y,w_in,b_in)
        final_w = final_w - (alpha*dj_dw)
        final_b = final_b - (alpha*dj_db)

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append(calculate_cost(x, y, final_w , final_b))
            p_history.append([final_w,final_b])
    return final_w,final_b,J_history,p_history

w,b,J_history,p_history = gradient_descent(x_train,y_train,0,0,2.0e-7,1000)

#print(J_history)
#print(J_history[0],J_history[-1],max(J_history))
#print(w,b)
#print(calculate_slope(x_train,y_train,w,b))
#print(calculate_cost(x_train,y_train,w,b))


#plt_gradients(x_train,y_train, calculate_cost, calculate_slope)

'''
fig, ax = plt.subplots(1,1, figsize=(12, 6))
plt_contour_wgrad(x_train, y_train, p_history, ax)
'''
# plot cost versus iteration 
''' 
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_history[:1000])
ax2.plot(14000 + np.arange(len(J_history[14000:])), J_history[14000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
'''
#plt.show()

