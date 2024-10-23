import streamlit as st


def page_project_conclusions_body():

    st.write("### Project Conclusions")

    st.write(
        f"The project was a success, with a deployed model capable of predicting the risk of heart disease on live inputted data."
    )
    
    st.success(
        f"#### Business Requirements\n\n"
        f"*Business Requirement 1* - This requirement was met through the use of an exploratory data analysis and various correlations and plots.\n"
        f"Through different methods it could be observed that the most highly correlating features were:\n"
        f"* Patient's chest pain type, exercise-induced angina, shape of ST slope on an ECG and ST depression on an ECG.\n\n"
        f"*Business Requirement 2* - This requirement was met through the use of a ML binary classification model.\n"
        f"* For recall on Heart Disease, the model scored 89% on the train set and 88% on the test set.\n"
        f"* For precision on No Heart Disease, the model scored 87% on the train set and 81% on the test set.\n"
    )

    st.info(
        f"#### Project Outcomes\n\n"
        f"The model performed well, although there is a discrepancy in the precision between the train and test sets which may be the result of some overfitting.\n\n"
        f"Hyperparameter optimisation proved more challenging than I expected, so I feel there is much more to be learned in this area.\n\n"
        f"Training the model on multiple datasets would be a good next step to see how different datasets behave, and to better understand how the data could be transformed."
    )