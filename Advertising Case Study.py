import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot

def MarvellousAdvertisementPredictor(data_path):

    data = pd.read_csv(data_path, index_col=0)

    print("Size of actual dataset ",len(data))

    feature_names = ['TV', 'radio', 'newspaper']

    print("Names of features ", feature_names)

    X = data[feature_names]

    y = data.sales

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/2)

    print("Size of training dataset ",len(X_train))

    print("Size of testing dataset ", len(X_test))

    linreg = LinearRegression()

    linreg.fit(X_train, y_train)

    y_pred = linreg.predict(X_test)

    print("Testing set")
    print(X_test)

    print("Result of testing ")
    print(y_pred)

    print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

def main():
    MarvellousAdvertisementPredictor("Advertising.csv")

if __name__ == "__main__":
    main()