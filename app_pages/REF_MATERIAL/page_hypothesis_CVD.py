import streamlit as st


def page_project_hypotheses_body():

    st.write("### Project Hypotheses")

    st.write(
        f"* [Hypothesis 1](#hypothesis-1)\n"
        f"* [Hypothesis 2](#hypothesis-2)\n"
        f"* [Hypothesis 3](#hypothesis-3)\n"
    )

    st.write(f"#### **Hypothesis 1**\n\n")
    st.info(
        f"* We suspect that the highest risk factors involved in heart disease are cholesterol and maximum heart rate.\n\n"
    )
    st.write(f"##### **Findings**:\n\n")
    st.info(
        f"* This hypothesis was incorrect.\n"
        f"* Chest pain type, exercise-induced angina, ST slope and ST depression were the most highly correlated features with heart disease.\n"
        f"* Maximum heart rate and cholesterol were found to have only very weak correlations using Pearson and Spearman correlations.\n"
        f"* Predictive power score found no correlation between heart disease and maximum heart rate or cholesterol.\n"
    )

    st.write(f"#### **Hypothesis 2**\n\n")
    st.warning(
        f"* We suspect that a successful prediction will rely on a large number of parameters.\n"
    )
    st.write(f"##### **Findings**:\n\n")
    st.warning(
        f"* The most important features from the model were found to be 'ST_Slope', 'ChestPainType', 'MaxHR', 'Age' and 'Cholesterol'.\n"
        f"* Despite the findings in the correlation studies, MaxHR, Age and Cholesterol were found to be necessary for the ML to make a prediction.\n"
        f"* 5 of the 11 features were necessary for a successful, so our hypothesis was correct."
    )

    st.write(f"#### **Hypothesis 3**\n\n")
    st.success(
        f"* We suspect that men over 50 with high cholesterol are the most at-risk patient group.\n\n"
    )
    st.write(f"##### **Findings**:\n\n")
    st.success(
        f"* A parallel plot to visualise trends in features was used to assess this hypothesis.\n"
        f"* It found that risk of heart disease increased with age.\n"
        f"* It also found that male patients were more prone to heart disease, however the dataset was imbalanced in favour of male patients.\n"
        f"* Cholesterol appeared to have no impact on heart disease, although this may be due to the data collected having a large number of '0' values.\n"
        f"* Therefore, our hypothesis was only partially correct."
    )