import plotly.express as px  # CHECK IF I NEED THIS!!!
import numpy as np  # CHECK IF I NEED THIS!!!
from feature_engine.discretisation import ArbitraryDiscretiser  # CHECK IF I NEED THIS!!!
from src.data_management import load_airplane_data   # CHECK IF I NEED THIS!!!

import streamlit as st
import pandas as pd
from src.data_management import load_airplane_data, load_pkl_file
# from src.machine_learning.predictive_analysis_ui import (        CHECK IF I NEED THIS!!!

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def page_overview_data_points_body():

    # load data
    df = load_airplane_data()

    # hard copied from churned customer study notebook
    vars_to_study = ['Contract', 'InternetService',  # CHANGE THIS!!!!!!!!!!!
                     'OnlineSecurity', 'TechSupport', 'tenure']  # CHANGE THIS!!!!!!!!!!!

    st.write("### General Aviation data set")
    st.info(
        f"* The client wants to understand the relationship between the Design and Performance parameters"
        f"* for Airplanes that falls into the category of General Aviation so that the client can learn how to"
        f" create quick conceptual aircraft designs drafts for performance specifications.")

    # inspect data
    if st.checkbox("Inspect Customer Base"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df.head(10))







import streamlit as st
import pandas as pd
from src.data_management import load_telco_data, load_pkl_file
from src.machine_learning.predictive_analysis_ui import (
# xxxxxxxxxxxxxxxxxxxxxxxxx    predict_churn,
# xxxxxxxxxxxxxxxxxxxxxxxxx    predict_tenure,
# xxxxxxxxxxxxxxxxxxxxxxxxx    predict_cluster)


def page_prospect_body():

    # load predict churn files
    version = 'v1'
    churn_pipe_dc_fe = load_pkl_file(
        f'outputs/ml_pipeline/predict_churn/{version}/clf_pipeline_data_cleaning_feat_eng.pkl')
    churn_pipe_model = load_pkl_file(
        f"outputs/ml_pipeline/predict_churn/{version}/clf_pipeline_model.pkl")
    churn_features = (pd.read_csv(f"outputs/ml_pipeline/predict_churn/{version}/X_train.csv")
                      .columns
                      .to_list()
                      )

    # load predict tenure files
    version = 'v1'
    tenure_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_tenure/{version}/clf_pipeline.pkl")
    tenure_labels_map = load_pkl_file(
        f"outputs/ml_pipeline/predict_tenure/{version}/label_map.pkl")
    tenure_features = (pd.read_csv(f"outputs/ml_pipeline/predict_tenure/{version}/X_train.csv")
                       .columns
                       .to_list()
                       )

    # load cluster analysis files
    version = 'v1'
    cluster_pipe = load_pkl_file(
        f"outputs/ml_pipeline/cluster_analysis/{version}/cluster_pipeline.pkl")
    cluster_features = (pd.read_csv(f"outputs/ml_pipeline/cluster_analysis/{version}/TrainSet.csv")
                        .columns
                        .to_list()
                        )
    cluster_profile = pd.read_csv(
        f"outputs/ml_pipeline/cluster_analysis/{version}/clusters_profile.csv")

    st.write("### Prospect Churnometer Interface")
    st.info(
        f"* The client is interested in determining whether or not a given prospect will churn. "
        f"If so, the client is interested to know when. In addition, the client is "
        f"interested in learning from which cluster this prospect will belong in the customer base. "
        f"Based on that, present potential factors that could maintain and/or bring  "
        f"the prospect to a non-churnable cluster."
    )
    st.write("---")

    # Generate Live Data
    # check_variables_for_UI(tenure_features, churn_features, cluster_features)
    X_live = DrawInputsWidgets()

    # predict on live data
    if st.button("Run Predictive Analysis"):
        churn_prediction = predict_churn(
            X_live, churn_features, churn_pipe_dc_fe, churn_pipe_model)

        if churn_prediction == 1:
            predict_tenure(X_live, tenure_features,
                           tenure_pipe, tenure_labels_map)

        predict_cluster(X_live, cluster_features,
                        cluster_pipe, cluster_profile)


def check_variables_for_UI(tenure_features, churn_features, cluster_features):
    import itertools

    # The widgets inputs are the features used in all pipelines (tenure, churn, cluster)
    # We combine them only with unique values
    combined_features = set(
        list(
            itertools.chain(tenure_features, churn_features, cluster_features)
        )
    )
    st.write(
        f"* There are {len(combined_features)} features for the UI: \n\n {combined_features}")


def DrawInputsWidgets():

    # load dataset
    df = load_telco_data()
    percentageMin, percentageMax = 0.4, 2.0

# we create input widgets only for 6 features
    col1, col2, col3, col4 = st.beta_columns(4)
    col5, col6, col7, col8 = st.beta_columns(4)

    # We are using these features to feed the ML pipeline - values copied from check_variables_for_UI() result

    # create an empty DataFrame, which will be the live data
    X_live = pd.DataFrame([], index=[0])

    # from here on we draw the widget based on the variable type (numerical or categorical)
    # and set initial values
    with col1:
        feature = "Contract"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique()
        )
    X_live[feature] = st_widget

    with col2:
        feature = "InternetService"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique()
        )
    X_live[feature] = st_widget

    with col3:
        feature = "MonthlyCharges"
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax,
            value=df[feature].median()
        )
    X_live[feature] = st_widget

    with col4:
        feature = "PaymentMethod"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique()
        )
    X_live[feature] = st_widget

    with col5:
        feature = "OnlineBackup"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique()
        )
    X_live[feature] = st_widget

    with col6:
        feature = "PhoneService"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique()
        )
    X_live[feature] = st_widget

    # st.write(X_live)

    return X_live
