import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def perform_individual_central_test_analysis_ilpd():
    datasets = {"ILPD": ["ilpd", "y"]}
    models = ["Logistic Regression", "Random Forests"]
    random_state = 42
    data = {}
    for key in datasets:
        for model in models:
            data[key + "_" + model] = []
            split_averages = []
            for client in ["coordinator", "participant1", "participant2", "participant3", "participant4"]:
                clients = []
                for i in range(1, 11):
                    train = pd.read_csv(f"classification/ILPD/data/5-split-10-cv/{client}/data/split_{i}/train.csv")
                    tests = []
                    for participant in ["coordinator", "participant1", "participant2", "participant3", "participant4"]:
                        tests.append(
                            pd.read_csv(
                                f"classification/ILPD/data/5-split-10-cv/{participant}/data/split_{i}/test.csv"))
                    test = pd.concat(tests)

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
                        clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
                    elif model == "Logistic Regression":
                        clf = LogisticRegression(penalty="none", solver='lbfgs', C=1e9, max_iter=10000,
                                                 fit_intercept=True)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                    clients.append(accuracy_score(y_true=y_test, y_pred=y_pred))

                split_averages.append(np.mean(clients))
            data[key + "_" + model] = split_averages

    df = pd.DataFrame(data)
    df = df.rename({"ILPD_Logistic Regression": "Logistic Regression (ILPD)",
                    "ILPD_Random Forests": "Random Forest (ILPD)"}, axis=1)
    df.to_csv(
        f'classification/ILPD/individual_central_test_results/individual_central_test_results.csv', index=False)
