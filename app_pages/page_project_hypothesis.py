import streamlit as st


def page_project_hypothesis_body():

    st.write("### Project Hypothesis and Validation")

    # Conclusions taken from "3A_airplane_engine_type_study" notebook
    st.success(
        "We suspect that airplanes with multiple engines are 'Higher, Further, Faster' and our proves this to be Correct!"
        "The correlation study in the 'Multi Engine Airplane Study' supports this hypothesis.\n\n"
        "The study of the airplane data showed a general performance increase for Multi Engine Airplanes in:\n"
        "- Service Ceiling (Hmax)\n"
        "- Range\n"
        "- Cruise and Max Speed\n\n"
        "But also an increase in:\n"
        "- Landing Speed (Vl)\n"
        "- Stall Speed (Vstall).\n\n"
        "This insight will enter into or clients conceptual design prediction tools."
    )
