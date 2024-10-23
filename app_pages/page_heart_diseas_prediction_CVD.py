import streamlit as st
import pandas as pd

from src.data_management import load_patient_data_raw, load_pkl_file
from src.machine_learning.classification import predict_live_heart_disease


def page_heartdisease_pred_body():

    version = "v2"
    dc_fe_pipeline = load_pkl_file(
        f"outputs/ml_pipeline/classification_model/{version}/data_cleaning_and_feat_engineering_pipeline.pkl"
    )
    model_pipeline = load_pkl_file(
        f"outputs/ml_pipeline/classification_model/{version}/classification_pipeline.pkl"
    )

    st.info(
        f"#### **Business Requirement 2**: Classification Model\n\n"
        f"* The client is interested in using patient data to predict whether or not a patient is at risk of heart disease.\n"
        f"* A machine learning model was built using a binary classification model with the following success metrics:\n"
        f"  * At least 75% recall for heart disease on train and test sets (no more than 25% missed positive predictions).\n"
        f"  * At least 70% precision for no heart disease (reducing the number of false positives)."
    )

    X_live = DrawInputsWidgets()

    if st.button("Run Predictive Analysis"):
        predict_live_heart_disease(
            X_live, dc_fe_pipeline, model_pipeline
        )


def DrawInputsWidgets():

    df = load_patient_data_raw()

    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7, col8 = st.columns(4)
    col9, col10, col11, col12 = st.columns(4)

    X_live = pd.DataFrame([], index=[0])

    with col1:
        feature = "Age"
        st_widget = st.number_input(
            label=feature,
            min_value=18,
            max_value=80,
            value=int(df[feature].median()),
            step=1,
        )
    X_live[feature] = st_widget

    with col2:
        feature = "Sex"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique(),
        )
    X_live[feature] = st_widget

    with col3:
        feature = "ChestPainType"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique(),
            help="TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic",
        )
    X_live[feature] = st_widget

    with col4:
        feature = "RestingBP"
        st_widget = st.number_input(
            label=feature,
            min_value=60,
            max_value=df[feature].max(),
            value=int(df[feature].median()),
            step=1,
        )
    X_live[feature] = st_widget

    with col5:
        feature = "Cholesterol"
        st_widget = st.number_input(
            label=feature,
            min_value=85,
            max_value=df[feature].max(),
            step=1,
        )
    X_live[feature] = st_widget

    with col6:
        feature = "FastingBS"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique(),
        )
    X_live[feature] = st_widget

    with col7:
        feature = "RestingECG"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique(),
            help="Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria"
        )
    X_live[feature] = st_widget

    with col8:
        feature = "MaxHR"
        st_widget = st.number_input(
            label=feature,
            min_value=60,
            max_value=202,
            step=1,
        )
    X_live[feature] = st_widget

    with col9:
        feature = "ExerciseAngina"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique(),
        )
    X_live[feature] = st_widget

    with col10:
        feature = "Oldpeak"
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min(),
            max_value=df[feature].max(),
            step=0.1,
            help="ST depression induced by exercise relative to rest"
        )
    X_live[feature] = st_widget

    with col11:
        feature = "ST_Slope"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique(),
            help="The slope of the peak exercise ST segment",
        )
    X_live[feature] = st_widget

    return X_live