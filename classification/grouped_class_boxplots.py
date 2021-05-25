import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def perform_grouped_boxplot_analysis_classification():
    lr = pd.read_csv(
        "classification/ILPD/centralized_results/Logistic Regression_sklearn.csv")
    lr_fc = pd.read_csv(
        "classification/ILPD/fc_results/evaluation/logistic_regression/coordinator/cv_evaluation.csv")
    rf_fc = pd.read_csv(
        "classification/ILPD/fc_results/evaluation/random_forest/coordinator/cv_evaluation.csv")
    rf = pd.read_csv(
        "classification/ILPD/centralized_results/Random Forests_sklearn.csv")

    lr_brca = pd.read_csv(
        "classification/BRCA/centralized_results/Logistic Regression_sklearn.csv")
    lr_fc_brca = pd.read_csv(
        "classification/BRCA/fc_results/evaluation/logistic_regression/coordinator/cv_evaluation.csv")
    rf_fc_brca = pd.read_csv(
        "classification/BRCA/fc_results/evaluation/random_forest/coordinator/cv_evaluation.csv")
    rf_brca = pd.read_csv(
        "classification/BRCA/centralized_results/Random Forests_sklearn.csv")

    metric = "accuracy"
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ILPD Dataset", "BRCA Dataset"))

    categories = ["Logistic Regression", "Random Forest"]
    x = []
    sklearn = []
    fc = []
    single = []
    single_new = []
    singles = pd.read_csv("classification/ILPD/individual_local_test_results/individual_local_test_results.csv")
    singles_new = pd.read_csv("classification/ILPD/individual_central_test_results/individual_central_test_results.csv")
    for category in categories:
        for i in range(10):
            x.append(category)
            if category == "Logistic Regression":
                sklearn.append(lr.loc[i, metric])
                fc.append(lr_fc.loc[i, metric])
            if category == "Random Forest":
                sklearn.append(rf.loc[i, metric])
                fc.append(rf_fc.loc[i, metric])
            if i <= 4:
                single.append(singles.loc[i, category + " (ILPD)"])
                single_new.append(singles_new.loc[i, category + " (ILPD)"])
            else:
                a = i - 5
                single.append(singles.loc[a, category + " (ILPD)"])
                single_new.append(singles_new.loc[a, category + " (ILPD)"])

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

    categories = ["Logistic Regression", "Random Forest"]
    x = []
    sklearn = []
    fc = []
    single = []
    single_new = []
    singles = pd.read_csv("classification/BRCA/individual_local_test_results/individual_local_test_results.csv")
    singles_new = pd.read_csv("classification/BRCA/individual_central_test_results/individual_central_test_results.csv")
    for category in categories:
        for i in range(10):
            x.append(category)
            if category == "Logistic Regression":
                sklearn.append(lr_brca.loc[i, metric])
                fc.append(lr_fc_brca.loc[i, metric])
            if category == "Random Forest":
                sklearn.append(rf_brca.loc[i, metric])
                fc.append(rf_fc_brca.loc[i, metric])
            if i <= 4:
                single.append(singles.loc[i, category + " (BRCA)"])
                single_new.append(singles_new.loc[i, category + " (BRCA)"])
            else:
                a = i - 5
                single.append(singles.loc[a, category + " (BRCA)"])
                single_new.append(singles_new.loc[a, category + " (BRCA)"])

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
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Box(
        y=single,
        x=x,
        name='Individual (Local Test Data)',
        marker_color='#a9a9a9',
        offsetgroup="D",
        showlegend=False
    ), row=1, col=2)

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

    fig.write_image("grouped_boxplot_analysis_classification.pdf")
    fig.write_image("grouped_boxplot_analysis_classification.png")
