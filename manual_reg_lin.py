import pandas as pd
import requests
import json
import numpy as np
import re
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def create_and_plot_manual_regression_model(X, Y):
    #Calculating the manual way

    # get the x-bar and y-bar
    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    #n is the number of rows
    n = len(X)
    print(n)

    # set the numerator and denominator as 0 first
    top = 0
    bottom = 0


    # type in the equations
    # for i in range(n) just means that for every row in total rows.
    # i typically starts at 0, and will run through this part of the code n(51) times
    for i in range(n):
        # you can check out the values of i in every for loop by adding this code:
        #print i
        top += (X[i] - mean_x) * (Y[i] - mean_y)
        bottom += (X[i] - mean_x) ** 2

    a = top / bottom
    b = mean_y - (a * mean_x)

    print(f"a is: {a}")
    # find the value of b (the constant/intercept) rounded up to 2 dec places
    print(f"b is: {b}")

    # time to plot it out
    y = b + a * X    

    plt.plot(X, y, color='red', label='Regression Line')
    plt.scatter(X, Y, c='blue', label='Actual data')

    plt.xlabel('City Population')
    plt.ylabel('Museum Visit')
    plt.legend()
    plt.show()