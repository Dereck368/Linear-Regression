import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

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

def main():
    df = get_DataFrame()
    tdf = get_test_df()

    # plots train data
    # scatter_plot_DataFrame(df)

    # get X vector
    x = df["Temperature"].to_numpy()

    # get y vector
    y = df["Icecream"].to_numpy()

    # print beta's
    beta0_hat, beta1_hat = linReg(x, y)
    print()
    print(f'β̂1 (Intercept): {beta0_hat} \nβ̂0 (Slope): {beta1_hat})')
    print()
    
    # get StatsModel
    res = stats_summary(x, y)
    print('-------')
    print("OlS Regression Summary")
    print('-------')
    print(res)

    # Can you conclude that there is a linear trend in the data? This is equivalent to ask whether you
    # can conclude that β1 is equal to zero or not. Justify your answer using statistical inference.

    # From this StatsModel we can conclude that their is a linear Trend in the data.
    # The reason for this is because our H₀ is β̂1 = 0 (No linear trend)
    # and our H₁ is β̂1 != 0 (there is a linear trend)
    #
    #

main()