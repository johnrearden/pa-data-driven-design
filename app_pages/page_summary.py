import streamlit as st

def page_summary_body():

    st.write("### Quick Project Summary")

    # text based on README file - "Dataset Content" section
    st.info(
        f"**Terms & Jargon**\n"
        f"* **General Aviation** is civil aviation other than large-scale passenger or freight operations.\n"
        f"* A **Categorical value** is a value that falls into distinct categories or groups. For example," 
        f" the Engine Type can be classified as piston, propjet, or jet.\n"
        f"* A **Continuous numeric value** is a value that can take any real number\n"
        f" (whole numbers and decimals). For example, the Cruise Speed can be 237,5 knots.\n\n"

        f"**Dataset**\n"

        f"* The dataset represents **857 Airplanes** in the category of **General Aviation** from"
        f" [Kaggle](https://www.kaggle.com/datasets/heitornunes/aircraft-performance-dataset-aircraft-bluebook?select=Airplane_Complete_Imputation.csv) "
        f"containing Meta data, 8 Design features, e.g. Dimensions and Weights "
        f"and 17 Performance features, e.g. Cruise Speed, Range and Rate of Climb. "
        f"Apart from three Categorical Design features the data set consists of continuous integers.")
