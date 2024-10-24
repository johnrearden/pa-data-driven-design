import streamlit as st
import pandas as pd


DATASET_DF = pd.read_csv(
    f"outputs/datasets/collection/HeartDiseasePrediction.csv").head(3)


def page_project_summary_body():

    st.write(
        f"* [Project Summary](#project-summary)\n"
        f"* [Project Dataset](#project-dataset)\n"
        f"* [Feature Terminology](#feature-terminology)\n"
        f"* [Business Requirements](#business-requirements)"
    )

    st.write("### Project Summary")

    st.write(
        f"Cardiovascular diseases are the number 1 cause of death globally, accounting for 31% of all deaths worldwide. People with cardiovascular disease or who are at high risk of disease need early detection and management.\n\n"
        f"A fictional organisation has requested a data practitioner to analyse a dataset of patients from a number of different hospitals in order to determine what factors can be attributed to a high risk of disease and whether patient data can accurately predict risk of heart disease.\n\n"
        f"* For further information, please visit and **read** the [project documentation](https://github.com/jfpaliga/CVD-predictor)."
    )

    st.info(
        f"#### **Project Dataset**\n\n"
        f"**Dataset**: A publically available dataset sourced from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data) was used for this project.\n\n"
        f"**Dataset Attributes**: The dataset contains 12 attributes, with 'HeartDisease' as the target.\n\n"
        f"**Dataset Observations**: The dataset contains a total of 918 observations."
    )

    st.dataframe(DATASET_DF)

    st.info(
        f"#### **Feature Terminology**\n\n"
        f"* **Age** - Age of the patient (years).\n"
        f"* **Sex** - Sex of the patient (M: Male, F: Female).\n"
        f"* **ChestPainType** - Category of chest pain: TA (typical angina), ATA (atypical angina), NAP (non-anginal pain), ASY (asymptomatic).\n"
        f"* **RestingBP** - Resting blood pressure (mm Hg).\n"
        f"* **Cholesterol** - Blood serum cholesterol (mm/dl).\n"
        f"* **FastingBS** - high diabetes risk (>120 mg/dl), low diabetes risk (otherwise).\n"
        f"* **RestingECG** - Resting electrocardiogram results (Normal: Normal, ST: having ST-T wave abnormality, LVH: showing probable or definite left ventricular hypertrophy).\n"
        f"* **MaxHR** - Maximum heart rate achieved.\n"
        f"* **ExerciseAngina** - Exercise-induced angina (Y: Yes, N: No).\n"
        f"* **Oldpeak** - Measure of ST segment depression on ECG (mm).\n"
        f"* **ST_Slope** - Slope of the peak exercise ST segment (Up: upsloping, Flat: flat, Down: downsloping).\n"
        f"* **HeartDisease** - Target feature (1: heart disease, 0: normal)."
    )

    st.success(
        f"#### **Business Requirements**\n\n"
        f"**Business Requirement 1** - The client is interested in which attributes correlate most closely with heart disease, ie what are the most common risk factors?\n\n"
        f"**Business Requirement 2** - The client is interested in using patient data to predict whether or not a patient is at risk of heart disease."
    )