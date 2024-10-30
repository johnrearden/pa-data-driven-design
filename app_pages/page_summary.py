import streamlit as st

def page_summary_body():

    st.write("### Quick Project Summary")

    # text based on README file - "Dataset Content" section
    st.info(
        f"**Terms & Jargon**\n"
        f"* **General Aviation** is civil aviation other than large-scale passenger or freight operations.\n"
        f"* A **Categorical value** is a value that falls into distinct categories, e.g." 
        f" the Engine Type can be classified as piston, propjet, or jet.\n"
        f"* A **Continuous numeric value** is a value that can take any real number\n"
        f" (whole numbers and decimals). For example, the Cruise Speed can be 237,5 knots.\n\n"

        f"**Dataset**\n"

        f"* The dataset represents **857 Airplanes** in the category of **General Aviation** from"
        f" [Kaggle](https://www.kaggle.com/datasets/heitornunes/aircraft-performance-dataset-aircraft-bluebook?select=Airplane_Complete_Imputation.csv) "
        f"containing Meta data, 8 Design features, e.g. Dimensions and Weights "
        f"and 17 Performance features, e.g. Cruise Speed, Range and Rate of Climb. "
        f"Apart from the meta data and three Categorical Design features the set consists of continuous integers.")



    # Link to README file, so the users can have access to full project documentation
    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/GustafEnebog/data-driven-design).")
    
    st.info(
        f"**Dataset**\n"
        f"**Business Requirement 1** - The client is interested in understanding the "
        f"relationships between the Design and Performance features in general and "
        f"which of these relationships have the greatest influence on each other.  \n"
        f"**Business Requirement 2** - The client is interested in categorizing the "
        f"airplanes based on engine type (jet, piston or propjet) using all features "
        f"and using supervised learning.  \n"
        f"**Business Requirement 3** - The client is interested in predicting the "
        f"necessary values that Wing Span (Design feature) need to take on in order "
        f"to reach certain performance targets.  \n"
        f"**Business Requirement 4** - The client is interested to see if any "
        f"“invisible” yet distinct airplane clusters (based on features) are hiding "
        f"in the data using unsupervised learning.")