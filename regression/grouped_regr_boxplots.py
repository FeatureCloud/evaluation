import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def perform_grouped_boxplot_analysis_regression():
    lr = pd.read_csv(
        "regression/diabetes/centralized_results/Linear Regression_sklearn.csv")
    lr_fc = pd.read_csv(
        "regression/diabetes/fc_results/evaluation/linear_regression/coordinator/cv_evaluation.csv")
    rf_fc = pd.read_csv(
        "regression/diabetes/fc_results/evaluation/random_forest/coordinator/cv_evaluation.csv")
    rf = pd.read_csv(
        "regression/diabetes/centralized_results/Random Forests_sklearn.csv")

    lr_boston = pd.read_csv(
        "regression/boston/centralized_results/Linear Regression_sklearn.csv")
    lr_fc_boston = pd.read_csv(
        "regression/boston/fc_results/evaluation/linear_regression/coordinator/cv_evaluation.csv")
    rf_fc_boston = pd.read_csv(
        "regression/boston/fc_results/evaluation/random_forest/coordinator/cv_evaluation.csv")
    rf_boston = pd.read_csv(
        "regression/boston/centralized_results/Random Forests_sklearn.csv")

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Diabetes Dataset", "Boston Dataset"))
    metric = "root_mean_squared_error"

    categories = ["Linear Regression", "Random Forest"]
    x = []
    sklearn = []
    fc = []
    single = []
    single_new = []
    singles = pd.read_csv("regression/diabetes/individual_local_test_results/individual_local_test_results.csv")
    singles_new = pd.read_csv("regression/diabetes/individual_central_test_results/individual_central_test_results.csv")
    for category in categories:
        for i in range(10):
            x.append(category)
            if category == "Linear Regression":
                sklearn.append(lr.loc[i, metric])
                fc.append(lr_fc.loc[i, metric])
            if category == "Random Forest":
                sklearn.append(rf.loc[i, metric])
                fc.append(rf_fc.loc[i, metric])
            if i <= 4:
                single.append(singles.loc[i, category + " (diabetes)"])
                single_new.append(singles_new.loc[i, category + " (diabetes)"])
            else:
                a = i - 5
                single.append(singles.loc[a, category + " (diabetes)"])
                single_new.append(singles_new.loc[a, category + " (diabetes)"])

    fig.add_trace(go.Box(
        y=sklearn,
        x=x,
        name='Centralized',
        marker_color='#F89A36',
        offsetgroup="A"
    ), row=1, col=1)

    fig.add_trace(go.Box(
        y=fc,
        x=x,
        name='Federated',
        marker_color='#19B7DF',
        offsetgroup="B"
    ), row=1, col=1)

    fig.add_trace(go.Box(
        y=single_new,
        x=x,
        name='Individual (Central Test Data)',
        marker_color='#65737e',
        offsetgroup="C",
        showlegend=True
    ), row=1, col=1)

    fig.add_trace(go.Box(
        y=single,
        x=x,
        name='Individual (Local Test Data)',
        marker_color='#a9a9a9',
        offsetgroup="D",
        showlegend=True
    ), row=1, col=1)


    categories = ["Linear Regression", "Random Forest"]
    x = []
    sklearn = []
    fc = []
    single = []
    single_new = []
    singles = pd.read_csv("regression/boston/individual_local_test_results/individual_local_test_results.csv")
    singles_new = pd.read_csv("regression/boston/individual_central_test_results/individual_central_test_results.csv")
    for category in categories:
        for i in range(10):
            x.append(category)
            if category == "Linear Regression":
                sklearn.append(lr_boston.loc[i, metric])
                fc.append(lr_fc_boston.loc[i, metric])
            if category == "Random Forest":
                sklearn.append(rf_boston.loc[i, metric])
                fc.append(rf_fc_boston.loc[i, metric])
            if i <= 4:
                single.append(singles.loc[i, category + " (boston)"])
                single_new.append(singles_new.loc[i, category + " (boston)"])
            else:
                a = i - 5
                single.append(singles.loc[a, category + " (boston)"])
                single_new.append(singles_new.loc[a, category + " (boston)"])

    fig.add_trace(go.Box(
        y=sklearn,
        x=x,
        name='Centralized',
        marker_color='#F89A36',
        offsetgroup="A",
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Box(
        y=fc,
        x=x,
        name='Federated',
        marker_color='#19B7DF',
        offsetgroup="B",
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Box(
        y=single_new,
        x=x,
        name='Individual (Central Test Data)',
        marker_color='#65737e',
        offsetgroup="C",
        showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Box(
        y=single,
        x=x,
        name='Individual (Local Test Data)',
        marker_color='#a9a9a9',
        offsetgroup="D",
        showlegend=False
    ), row=1, col=2)

    fig.update_yaxes(range=[0, 20], row=1, col=2)

    fig.update_layout(
        yaxis_title=metric.replace("_", " ").capitalize(),
        boxmode='group',
        template="simple_white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5),
    )

    fig.write_image("grouped_boxplot_analysis_regression.pdf")
    fig.write_image("grouped_boxplot_analysis_regression.svg")
    fig.write_image("grouped_boxplot_analysis_regression.png")
