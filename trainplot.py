import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from sklearn.feature_selection import SequentialFeatureSelector
import numpy as np

def get_DataFrame():
    columns = ["Temperature", "Icecream"]
    df = pd.read_fwf('train.txt', header=None, names=columns)
    return df

def get_test_df():
    columns = ["Temperature", "Icecream"]
    df = pd.read_fwf('test.txt', header=None, names=columns)
    return df

def scatter_plot_DataFrame(df):    
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1,1)

    ax.scatter(df["Temperature"], df["Icecream"], color="green", s=20, marker='o', label='Train Data')
    ax.set_xlabel("Temperature")
    ax.set_ylabel("# of Ice Cream Sold")
    ax.set_title("Train Data")
    ax.legend()

    plt.show()

def linReg(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train.reshape(-1, 1), y_train)

    beta0_hat = model.intercept_
    beta1_hat = model.coef_[0]

    return beta0_hat, beta1_hat

def stats_summary(X, y):
    constant_x = sm.add_constant(X)

    model = sm.OLS(y,constant_x)
    results = model.fit()

    return results.summary()

# fix discription and variable names of function
def model(df):
    x = df["Temperature"].to_numpy()
    y = df["Icecream"].to_numpy()

    epsilon = 1e-8

    # different features
    X = pd.DataFrame({
    'x': x,
    'cos_x': np.cos(x),
    'log_x': np.log(x + epsilon), #avoid log(0)
    'cos_4x': np.cos(4 * x),
    'sin_3x': np.sin(3 * x),
    'sin_5x': np.sin(5 * x),
    'sin_2x_cos_2x': np.sin(2 * x) * np.cos(2 * x)
    })

    #run selction on num of features get list of each M0 M1 etc...
    print(X)
    p = X.shape[1] - 1
    #help with indexing error without '- 1'
    # ok so we do 7 choose 6 features to add that's why it is -1
    # only problem here is if if we want to choose 6 features wouldn't it get rid of 
    # x
        # if it does get rid of x does that mean that x wasn't an optimal feature
        # for a solution.

    best_features = []
    best_adjusted_r2 = -np.inf

    for k in range(1, p+1):
        model = LinearRegression()

        selector = SequentialFeatureSelector(model, n_features_to_select=k, direction='forward', scoring='r2')
        selector.fit(X, y)
        
        # gets the selected features
        features_selected = selector.get_support(indices=True)
        X_train = X.iloc[:, features_selected]
        
        model.fit(X_train, y)
        r2 = model.score(X_train, y)

        n = X.shape[0]
        d = X_train.shape[1]

        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - d - 1))

        if adjusted_r2 > best_adjusted_r2:
            best_adjusted_r2 = adjusted_r2
            best_features = features_selected

    print(X.columns[best_features])
    print(adjusted_r2)

    # create return method
    # graph adjusted r2 of each model
        #to do this create list for models then graph each one

def main():
    df = get_DataFrame()

    # plots train data
    # scatter_plot_DataFrame(df)

    # get X vector
    x = df["Temperature"].to_numpy()

    # get y vector
    y = df["Icecream"].to_numpy()

    # print beta's
    beta0_hat, beta1_hat = linReg(x, y)
    # print()
    # print(f'β̂0 (Slope): {beta0_hat}) \nβ̂1 (Intercept): {beta1_hat} )')
    # print()

    model(df)
    
    # get StatsModel
    res = stats_summary(x, y)
    # print(res)


main()