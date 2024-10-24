import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from src.data_management import load_pkl_file
from src.machine_learning.evaluate_clf import clf_performance


def page_model_performance_body():

    version = "v2"
    dc_fe_pipeline = load_pkl_file(
        f"outputs/ml_pipeline/classification_model/{version}/data_cleaning_and_feat_engineering_pipeline.pkl"
    )
    model_pipeline = load_pkl_file(
        f"outputs/ml_pipeline/classification_model/{version}/classification_pipeline.pkl"
    )
    feat_importance = plt.imread(
        f"outputs/ml_pipeline/classification_model/{version}/features_importance.png"
    )
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/classification_model/{version}/X_train.csv"
    )
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/classification_model/{version}/X_test.csv"
    )
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/classification_model/{version}/y_train.csv"
    )
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/classification_model/{version}/y_test.csv"
    )

    st.write("### ML Pipeline: Binary Classification")

    st.info(
        f"The model success metrics are:\n"
        f"* At least 75% recall for heart disease (the model minimises the chances of missing a posititve diagnosis).\n\n"
        f"The model will be considered a failure if:\n"
        f"* The model fails to achieve 75% recall for heart disease.\n"
        f"* The model fails to achieve 70% precision for no heart disease (false positives).\n"
    )

    st.write("---")
    st.write(f"#### ML Pipelines")
    st.write(f"For this model there were 2 ML Pipelines arrange in series:\n")

    st.write(f"* The first pipeline is responsible for data cleaning and feature engineering.\n")
    st.write(dc_fe_pipeline)

    st.write(f"* The second pipeline is responsible for feature scaling and modelling.\n")
    st.write(model_pipeline)

    st.write("---")
    st.write(f"#### Feature Importance")
    st.write(f"* The most important features used for training the model were as follows:\n")
    st.write(X_train.columns.to_list())
    st.image(feat_importance)

    st.write("---")
    st.write(f"#### Model Performance")
    st.success(
        f"The model passed the acceptance criteria with the following metrics:\n"
        f"* Recall on Heart Disease: 89% on train set, 88% on test set.\n"
        f"* Precision on No Heart Disease: 87% on train set, 81% on test set."
        )

    st.write("---")

    clf_performance(X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    pipeline=model_pipeline,
                    label_map=["No Heart Disease", "Heart Disease"])