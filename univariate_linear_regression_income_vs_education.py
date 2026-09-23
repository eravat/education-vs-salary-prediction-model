import pandas as pd
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
    final_w = w_in
    final_b = b_in
    for i in range(number_of_iterations):
        dj_dw, dj_db = calculate_slope(x,y,w_in,b_in)
        final_w = final_w - (alpha*dj_dw)
        final_b = final_b - (alpha*dj_db)
    return final_w,final_b

w,b = gradient_descent(x_train,y_train,0,0,3.0e-7,20000)
#print(w,b)
#print(calculate_slope(x_train,y_train,w,b))
#print(calculate_cost(x_train,y_train,w,b))