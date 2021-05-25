import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


def perform_centralized_analysis_ilpd():
    def create_eval_dataframe(accs, f1s, mccs, precs, recs, runs):
        scores = [accs, f1s, mccs, precs, recs, runs]
        cols = ["accuracy", "f1-score", "mcc", "precision", "recall", "runtime"]

        df = pd.DataFrame(data=scores).transpose()
        df.columns = cols

        return df

    def plot_boxplots(df, title):
        fig = go.Figure()
        fig.add_trace(go.Box(y=df["accuracy"], quartilemethod="linear", name="Accuracy"))
        fig.add_trace(go.Box(y=df["precision"], quartilemethod="linear", name="Precision"))
        fig.add_trace(go.Box(y=df["recall"], quartilemethod="linear", name="Recall"))
        fig.add_trace(go.Box(y=df["f1-score"], quartilemethod="linear", name="f1-score"))
        fig.add_trace(go.Box(y=df["mcc"], quartilemethod="linear", name="MCC"))
        fig.update_layout(title=title)
        fig.update_yaxes(range=[0.5, 1])

        return fig

    models = ["Logistic Regression", "Random Forests"]

    random_state = 42
    label_col = "y"

    for model in models:
        accs = []
        f1s = []
        mccs = []
        precs = []
        recs = []
        runs = []

        # data = pd.read_csv(datasets[dataset_key][0])
        # target = data.loc[:, datasets[dataset_key][1]]
        # data = data.drop(datasets[dataset_key][1], axis=1)
        splits = []
        input_reading_runtime = []

        for i in range(1, 11):
            print(i)
            start_time = time.time()
            splits.append(pd.read_csv(f'data/10-cv-splits/split_{i}.csv'))
            input_reading_runtime.append(time.time() - start_time)

        for i in range(len(splits)):
            print(i)
            start_time = time.time()
            test_split = splits[i]
            train_splits = [i for i in splits.copy() if not i.equals(test_split)]
            X_train = pd.concat(train_splits)
            y_train = X_train.loc[:, label_col]
            X_train = X_train.drop(label_col, axis=1)
            X_test = test_split.drop(label_col, axis=1)
            y_test = test_split.loc[:, label_col]

            standardizer = StandardScaler()
            standardizer.fit(X_train)
            X_train = standardizer.transform(X_train)
            X_test = standardizer.transform(X_test)

            clf = None
            if model == "Logistic Regression":
                clf = LogisticRegression(penalty="none", solver='lbfgs', C=1e9, max_iter=10000, fit_intercept=True)
            elif model == "Random Forests":
                clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            accs.append(acc)
            f1 = f1_score(y_test, y_pred)
            f1s.append(f1)
            mcc = matthews_corrcoef(y_test, y_pred)
            mccs.append(mcc)
            prec = precision_score(y_test, y_pred)
            precs.append(prec)
            rec = recall_score(y_test, y_pred)
            recs.append(rec)
            run = time.time() - start_time + input_reading_runtime[i]
            runs.append(run)

        mean_runtime = round(np.mean(runs), 3)
        std_runtime = round(np.std(runs), 3)

        score_df = create_eval_dataframe(accs, f1s, mccs, precs, recs, runs)
        plt = plot_boxplots(score_df, title=f'ILPD: {model}')
        plt.show()
        score_df.to_csv(f'centralized_results/{model}_sklearn.csv', index=False)
        plt.write_image(f'centralized_results/{model}_sklearn.png')
        plt.write_image(f'centralized_results/{model}_sklearn.svg')
        plt.write_image(f'centralized_results/{model}_sklearn.pdf')
