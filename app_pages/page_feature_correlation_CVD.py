import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import ppscore as pps
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import dc_no_encoding_pipeline, one_hot_encode
from src.utils import dc_all_cat_pipeline, map_discretisation


raw_df = pd.read_csv(f"outputs/datasets/collection/HeartDiseasePrediction.csv")
clean_df = dc_no_encoding_pipeline(raw_df)


def correlation(df, method):
    """
    Perform one hot encoding on a Dataframe and
    return the absolute values of correlation to
    the target in descending order
    """

    ohe_df = one_hot_encode(df)
    corr = ohe_df.corr(method=method)["HeartDisease"].sort_values(
        key=abs, ascending=False)[1:].head(10)

    return corr


def heatmap_pps(df, threshold, figsize=(18, 14), font_annot=10):
    """
    Calculate a PPS matrix from a Dataframe then plot
    a heatmap using a threshold
    """

    pps_matrix_raw = pps.matrix(df)
    pps_matrix_df = pps_matrix_raw.filter(["x", "y", "ppscore"]).pivot(
        columns="x", index="y", values="ppscore")

    if len(pps_matrix_df.columns) > 1:
        mask = np.zeros_like(pps_matrix_df, dtype=bool)
        mask[abs(pps_matrix_df) < threshold] = True

        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(pps_matrix_df, annot=True,
                         annot_kws={"size": font_annot},
                         mask=mask, cmap='rocket_r', linewidth=0.05,
                         linecolor='lightgrey')

        plt.ylim(len(pps_matrix_df.columns), 0)
        st.pyplot(fig)


def plot_categorical(df, col):
    """
    Plot a count plot of a given column in a Dataframe
    """

    fig, ax = plt.subplots(figsize=(14, 8))
    ax = sns.countplot(data=df, x=col,
                       hue="HeartDisease",
                       order=df[col].value_counts().index)

    plt.xticks(rotation=90)
    plt.title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)


def parallel_plot(df):
    """
    Plot a categorical parallel plot from a Dataframe
    """

    pplot_df = dc_all_cat_pipeline(df)
    pplot_df = map_discretisation(pplot_df)

    columns = pplot_df.drop(["HeartDisease"], axis=1).columns.to_list()
    fig = px.parallel_categories(pplot_df, color="HeartDisease",
                                 dimensions=columns,
                                 color_continuous_scale="bluered",)
    st.plotly_chart(fig)


def page_feat_correlation_body():

    st.write("### Feature Correlation Study")

    st.write(
        f"* [Business Requirement and Dataset](#business-requirement-1-data-visualisation-and-correlation-study)\n"
        f"* [Summary of Correlation Analysis](#summary-of-correlation-analysis)\n"
        f"* [Summary of PPS Analysis](#summary-of-pps-analysis)\n"
        f"* [Analysis of Most Correlated Features](#analysis-of-most-correlated-features)\n"
        f"* [Feature Relationships](#feature-relationships)\n"
        f"* [Conclusions](#conclusions)\n"
        )

    st.info(
        f"#### **Business Requirement 1**: Data Visualisation and Correlation Study\n\n"
        f"* We need to perform a correlation study to determine which features correlate most closely to the target.\n"
        f"* A Pearson's correlation will indicate linear relationships between numerical variables.\n"
        f"* A Spearman's correlation will measure the monotonic relationships between variables.\n"
        f"* A Predictive Power Score study can also be used to determine relationships between attributes regardless of data type (6/11 features are categorical).\n"
    )

    if st.checkbox("Inspect heart disease dataset"):
        st.dataframe(raw_df.head(5))
        st.write(f"The dataset contains {raw_df.shape[0]} observations with {raw_df.shape[1]} attributes.")

    st.write("---")

    st.write(
        f"#### **Summary of Correlation Analysis**\n"
        f"* Correlations within the dataset were analysed using Spearman and Pearson correlations followed by a Predictive Power Score (PPS) analysis.\n"
        f"* For the Spearman and Pearson correlations, all categorical features from the cleaned dataset were one hot encoded.\n"
        f"* Both methods found the same correlation between features and target.\n"
    )

    if st.checkbox("View Pearson correlation results"):
        st.write(correlation(clean_df, method="pearson"))
    if st.checkbox("View Spearman correlation results"):
        st.write(correlation(clean_df, method="spearman"))

    st.write("---")

    st.write(
        f"#### **Summary of PPS Analysis**\n"
        f"* The PPS analyis also indicated the same features had the greatest correlation to the target.\n"
        f"* These features were: ChestPainType, ExerciseAngina and ST_Slope.\n"
        f"* The PPS analysis also indicated that the Oldpeak feature had some weak correlation to the target.\n"
    )

    if st.checkbox("View PPS heatmap"):
        heatmap_pps(clean_df, 0.15)

    st.write("---")

    st.write(
        f"#### **Analysis of Most Correlated Features**\n"
        f"* An asymptomatic (ASY) chest pain type is typically associated with heart disease.\n"
        f"* Exercise-induced angina is typically associated with heart disease, although a significant portion of patients without exercise-induced angina also had heart disease.\n"
        f"* An ST depression of >1.5 mm is typically associated with heart disease, and roughly 50% of patients with an ST depression between 0 and 1.5 had heart disease.\n"
        f"* A flat ST slope is typically associated with heart disease. A down ST slope is generally more related to heart disease as well, however there is less data on this.\n"
    )

    feature_distribution = st.selectbox(
        "Select feature to view distribution:",
        ("ChestPainType", "ExerciseAngina", "Oldpeak", "ST_Slope")
    )

    plot_categorical(clean_df, feature_distribution)

    st.write("---")

    st.write(
        f"#### **Feature Relationships**\n"
        f"* A parallel plot was used to further assess the relationships between features.\n"
        f"* The plot shows imbalance in some of the features, e.g. sex of the patients.\n"
        f"* Despite this, some general trends can be observed:\n"
        f"  * Risk of heart disease increases with age.\n"
        f"  * Male patients tend to be at greater risk.\n"
        f"  * High diabetes risk patients were more prone to heart disease.\n"
    )

    if st.checkbox("View parallel plot"):
        parallel_plot(raw_df)

    st.write("---")

    st.success(
        f"#### **Conclusions**\n\n"
        f"* We stated in **hypothesis 1** that we expected cholesterol and maximum heart rate to be the greatest risk factors related to heart disease.\n"
        f"* The findings from the exploratory data analysis found this hypothesis to be **incorrect.**\n"
        f"* From the correlation studies, it was found that the most highly correlating features were:\n"
        f"  * A patient's chest pain type.\n"
        f"  * Chest pain (angina) induced from exercise.\n"
        f"  * An ST depression of >1.5 mm on an ECG.\n"
        f"  * A flat or downward sloping ST slope on an ECG.\n"
    )