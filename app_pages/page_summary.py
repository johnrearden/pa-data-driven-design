import streamlit as st


def page_summary_body():
    st.write("### Quick Project Summary")

    # text based on README file - "Dataset Content" section
    st.info(
        f"**Terms**\n"
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


    # copied from README file - "Business Requirements" section
    st.success(
        f"The project has 7 business requirements:\n"
        f"* 1 - The client is interested in having the data"
        f" set concretized to the reality of actual airplanes in terms of bounds"
        f" and mean etc. as well as understanding the relationships between the"
        f" Design and Performance features in general and which of these"
        f" relationships are having the greatest influence on each other.  \n"  

        f"* 2 - The client wants to test the premise that"
        f" airplanes with Multiple Engines are “Higher, Further, Faster”.  \n"  

        f"* 3 - The client is interested in evaluating"
        f" the “strength and weakness”-profile for two of their main competitors"
        f" Piper and Cessna by evaluating the differences in performance for"
        f" different features.  \n"  

        f"* 4 - The client is interested in predicting"
        f" the necessary values that an airplanes Wing Span (Design feature)"
        f" need to take on in order to reach certain performance targets.  \n"  

        f"* 5 - The client is interested to see if"
        f" any “invisible” yet distinct airplane clusters (based on features)"
        f" are hiding in the data using unsupervised learning.  \n"  

        f"* 6 - The client (who currently investigates"
        f" a new design for a high altitude loiter airplane) is interested"
        f" to establish if their assumption that the max altitude (ceiling)"
        f" an airplane can fly at is heavily dependent on span (larger span,"
        f" higher altitude).  \n"  

        f"* 7 -  The client wishes to refine their"
        f" design tools and validate the classic Breguet Range equation,"
        f" which, among other things, says that the distance an airplane"
        f" can fly is linearly dependent on a weight ratio that is getting"
        f" larger when, the portion of the airplanes weight that is made"
        f" up by fuel, is getting larger.  \n")  