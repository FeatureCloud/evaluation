import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, max_error, mean_squared_error, median_absolute_error
from sklearn.preprocessing import StandardScaler


def perform_centralized_analysis_boston():
    def create_eval_dataframe(maes, maxs, rmses, mses, medaes):
        scores = [maes, maxs, rmses, mses, medaes]
        cols = ["mean_absolut_error", "max_error", "root_mean_squared_error", "mean_squared_error",
                "median_absolut_error"]

        df = pd.DataFrame(data=scores).transpose()
        df.columns = cols

        return df

    def plot_boxplots(df, title):
        fig = go.Figure()
        fig.add_trace(
            go.Box(y=df["mean_absolut_error"].map(np.log2), quartilemethod="linear", name="Mean Absolut Error"))
        fig.add_trace(go.Box(y=df["max_error"].map(np.log2), quartilemethod="linear", name="Max Error"))
        fig.add_trace(go.Box(y=df["root_mean_squared_error"].map(np.log2), quartilemethod="linear",
                             name="Root Mean Squared Error"))
        fig.add_trace(
            go.Box(y=df["median_absolut_error"].map(np.log2), quartilemethod="linear", name="Median Absolut Error"))
        fig.update_layout(title=title, yaxis_title='Log2')

        return fig

    models = ["Linear Regression", "Random Forests"]

    random_state = 42
    label_col = "label"

    for model in models:
        maes = []
        maxs = []
        rmses = []
        mses = []
        medaes = []
        runs = []

        splits = []

        input_reading_runtime = []
        for i in range(1, 11):
            start_time = time.time()
            splits.append(pd.read_csv(f'regression/boston/data/10-cv-splits/split_{i}.csv'))
            input_reading_runtime.append(time.time() - start_time)

        for i in range(len(splits)):
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

            regr = None
            if model == "Linear Regression":
                regr = LinearRegression()
            elif model == "Random Forests":
                regr = RandomForestRegressor(n_estimators=100, random_state=random_state)
            regr.fit(X_train, y_train)
            y_pred = regr.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            maes.append(mae)
            m = max_error(y_test, y_pred)
            maxs.append(m)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            rmses.append(rmse)
            mse = mean_squared_error(y_test, y_pred)
            mses.append(mse)
            medae = median_absolute_error(y_test, y_pred)
            medaes.append(medae)
            run = (time.time() - start_time) + input_reading_runtime[i]
            runs.append(run)

        score_df = create_eval_dataframe(maes, maxs, rmses, mses, medaes)
        plt = plot_boxplots(score_df, title=f'Boston: {model}')
        score_df.to_csv(f'regression/boston/centralized_results/{model}_sklearn.csv', index=False)
        plt.write_image(f'regression/boston/centralized_results/{model}_sklearn.pdf')
