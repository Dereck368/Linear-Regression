import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import r2_score, root_mean_squared_error
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

def suggested_features(x):
    epsilon = 1e-8

    # different features
    X = pd.DataFrame({
    'x': x,
    'cos(x)': np.cos(x),
    'log(x)': np.log(x + epsilon), #avoid log(0)
    'cos(4x)': np.cos(4 * x),
    'sin(3x)': np.sin(3 * x),
    'sin(5x)': np.sin(5 * x),
    'sin(2x)cos(2x)': np.sin(2 * x) * np.cos(2 * x)
    })

    return X

def select_model(x_train, y_train):
    X = suggested_features(x_train)
    print(X)
    
    p = X.shape[1] - 1

    best_features = []
    best_adjusted_r2 = -np.inf

    adj_r2_values = []
    best_model = LinearRegression()

    for k in range(1, p+1):
        model = LinearRegression()

        selector = SequentialFeatureSelector(model, n_features_to_select=k, direction='forward', scoring='r2')
        selector.fit(X, y_train)
        
        # gets the selected features
        features_selected = selector.get_support(indices=True)
        X_train = X.iloc[:, features_selected]
        
        #calculate adjusted r2
        model.fit(X_train, y_train)
        r2 = model.score(X_train, y_train)

        n = X.shape[0]
        d = X_train.shape[1]

        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - d - 1))

        adj_r2_values.append(adjusted_r2)

        #compares adjusted_r2 between models
        if adjusted_r2 > best_adjusted_r2:
            best_adjusted_r2 = adjusted_r2
            best_features = features_selected
            best_model = model

    print(X.columns[best_features])
    print(best_adjusted_r2)
 
    # plot
    # plt.figure(figsize=(10, 6))
    # plt.scatter(np.arange(len(adj_r2_values)), adj_r2_values, s=10, color='purple')
    # plt.plot(np.arange(len(adj_r2_values)), adj_r2_values, color='purple') 
    # plt.xlabel("Models")
    # plt.ylabel("Adjusted R²")
    # plt.title("Adjusted R² Across Forward Selection Models")
    # plt.xticks(rotation=45)
    # plt.show()
    # plt.savefig("adjustedR2.png")
    return X.columns[best_features] , best_model

def lasso_reg(X_train, y_train):
    lasso_cv = LassoCV(cv=5)

    lasso_cv.fit(X_train, y_train)

    #optimal alpha 
    alpha = lasso_cv.alpha_

    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)

    n = X_train.shape[0]  # number of samples
    p = X_train.shape[1]  # number of predictors

    lasso_pred = lasso.predict(X_train)
    lasso_r2 = r2_score(y_train, lasso_pred)
    lasso_adj_r2 = 1 - (1 - lasso_r2) * (n - 1) / (n - p - 1)

    print(lasso_adj_r2)

    # plot
    # plt.scatter(X_train['x'], y_train, color='green', s=10, label='Training data')
    # plt.plot(X_train['x'], lasso_pred, color='red', linewidth=2, label='Lasso regression line')
    # plt.xlabel("Feature")
    # plt.ylabel("Target")
    # plt.title("Lasso Regression Line on Training Data")
    # plt.legend()
    # plt.show()

    return lasso_adj_r2, alpha, lasso

def ridge_reg(X_train, y_train):
    ridge_cv = RidgeCV(cv=5)
    ridge_cv.fit(X_train, y_train)

    alpha = ridge_cv.alpha_

    n = X_train.shape[0]  # number of samples
    p = X_train.shape[1]  # number of predictors

    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)

    ridge_pred = ridge.predict(X_train)
    ridge_r2 = r2_score(y_train, ridge_pred)
    ridge_adj_r2 = 1 - (1 - ridge_r2) * (n - 1) / (n - p - 1)

    print(ridge_adj_r2)

    # plot
    # plt.scatter(X_train['x'], y_train, color='green', s=10, label='Training data')
    # plt.plot(X_train['x'], ridge_pred, color='red', linewidth=2, label='Ridge regression line')
    # plt.xlabel("Feature")
    # plt.ylabel("Target")
    # plt.title("Ridge Regression Line on Training Data")
    # plt.legend()
    # plt.show()

    return ridge_adj_r2, alpha, ridge

def predictions_on_test_data(X_test, y_test, model, model_name="Model"):
    X = suggested_features(X_test)
    y_pred = model.predict(X)

    n = X.shape[0]  # number of samples
    p = X.shape[1]  # number of predictors

    mse = root_mean_squared_error(y_test, y_pred) ** 2
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print(f"\n{model_name} performance on the Test Data:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Adjusted R squared: {adj_r2:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(X['x'], y_test, color='green', s=10, label='Actual Test Data')
    plt.plot(X['x'], y_pred, color='red', linewidth=2, label=f'{model_name} Predictions')
    plt.xlabel('Temperature')
    plt.ylabel('# of Ice Cream Sold')
    plt.title(f'{model_name} Predictions v Actual Test Data')
    plt.legend()
    plt.show()

def linear_reg_predictions_on_test_data(X_test, y_test, X_train, y_train):
    X = suggested_features(X_test)
    features, model = select_model(X_train, y_train)
    X = X[features]
    y_pred = model.predict(X)

    n = X.shape[0]  # number of samples
    p = X.shape[1]  # number of predictors
    
    mse = root_mean_squared_error(y_test, y_pred) ** 2
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print("Lienar Regression performance on the Test Data:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Adjusted R squared: {adj_r2:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(X['x'], y_test, color='green', s=10, label='Actual Test Data')
    plt.plot(X['x'], y_pred, color='red', linewidth=2, label='Linear Regression Predictions')
    plt.xlabel('Temperature')
    plt.ylabel('# of Ice Cream Sold')
    plt.title('Linear Predictions v Actual Test Data')
    plt.legend()
    plt.show()

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

    # for best model selection
    # select_model(df)

    new_X = suggested_features(x)

    l, la, ridge = lasso_reg(new_X, y)
    r, ra, lasso = ridge_reg(new_X, y)

    # print(f'Lasso optimal alpha: {la}')
    # print(f'Lasso adjusted R2 value: {l}')

    # print(f'Ridge optimal alpha: {ra}')
    # print(f'Ridge adjusted R2 value: {r}')
    
    # get StatsModel
    # res = stats_summary(x, y)
    # # print(res)
    # select_model(x,y)

    tdf = get_test_df()
    X_test= tdf['Temperature']
    y_test = tdf['Icecream']
    linear_reg_predictions_on_test_data(X_test, y_test, x, y)
    predictions_on_test_data(X_test, y_test, ridge, 'Ridge Regression')
    predictions_on_test_data(X_test, y_test, lasso, 'Lasso Regression')

main()