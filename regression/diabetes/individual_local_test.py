import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def perform_individual_local_test_analysis_diabetes():
    datasets = {"diabetes": ["diabetes", "target"]}
    models = ["Linear Regression", "Random Forests"]
    random_state = 42
    data = {}
    for key in datasets:
        for model in models:
            data[key + "_" + model] = []
            split_averages = []
            for client in ["coordinator", "participant1", "participant2", "participant3", "participant4"]:
                clients = []
                for i in range(1, 11):
                    train = pd.read_csv(
                        f"data/5-split-10-cv/{client}/data/split_{i}/train.csv")
                    test = pd.read_csv(f"data/5-split-10-cv/{client}/data/split_{i}/test.csv")
                    X_train = train.drop(datasets[key][1], axis=1)
                    y_train = train.loc[:, datasets[key][1]]

                    X_test = test.drop(datasets[key][1], axis=1)
                    y_test = test.loc[:, datasets[key][1]]

                    standardizer = StandardScaler()
                    standardizer.fit(X_train)
                    X_train = standardizer.transform(X_train)
                    X_test = standardizer.transform(X_test)

                    clf = None
                    if model == "Random Forests":
                        clf = RandomForestRegressor(n_estimators=100, random_state=random_state)
                    elif model == "Linear Regression":
                        clf = LinearRegression()
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                    clients.append(mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False))

                split_averages.append(np.mean(clients))
            data[key + "_" + model] = split_averages

        df = pd.DataFrame(data)
        df = df.rename({"diabetes_Linear Regression": "Linear Regression (diabetes)",
                        "diabetes_Random Forests": "Random Forest (diabetes)"}, axis=1)

        df.to_csv("individual_local_test_results/individual_local_test_results.csv", index=False)
