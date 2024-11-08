import streamlit as st


def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"*  **1**"
        f" - We suspect Airplanes with multiple engines are 'Higher, Further, Faster': **Correct**. \n"  
        f" The study on: 'Multi Engine Airplane Study' supports this. \n"  
        f"   - Validation: The correlation model shows that"
        f" a Multi Engined Airplane on average has a higher:"
        f" Hmax, Vcruise, Vl, Vmax and Vstall than a Single Engined Airplane. \n\n\n"

        f"* **2** (BR1)"
        f"  - We propose that it is possible to create a simple regression function,"
        f" based on historical Piper and Cessna airplane design and performance data, that can"
        f" visualize the differences between the two manufacturers in different"
        f" performance features along the range where both manufacturers are"
        f" represented with airplanes: **Correct**."  
        f" Usage of the 'Regression Playground' with the selection of 'Piper vs. Cessna' under"
        f" the Filter Option supports this. \n"  
        f"  - Validation: The implemented algorithms with one regression lines for each manufacturer"
        f" Shows that the manufacturers performance varies from each other for different features"
        f" and for different intervalls of each feature that can be compared with each other. \n\n\n"  

        f"* **3** (BR2)"
        f"  - We propose that it is possible to create a model, based on historical" 
        f" airplane design and performance data, that predicts the Wing Span" 
        f" of an airplane with a reasonable accuracy: **Correct**."         
        f" Usage of the 'Regression Playground' with the selection of 'All airplanes' under"
        f" the Filter Option supports this. \n"  
        f"  - Validation: The implemented algorithms in the user interface shows"
        f" that the user can predict the target variable, Wing Span, by inputting"
        f" desired airplane performance and design features."
        f" that can be compared with each other and that the relative error (RE) is less than 5.97%. \n\n\n"  

        f"* **4** (BR3)"
        f"  - We suspect that the data hides a few but not many distinct clusters"
        f" of airplanes within the data set: **Not quite correct**." 
        f" The study on: 'page_ml_cluster' explane why this is not quite correct. \n"  
        f"  - Validation: Our cluster model found only two trivial clusters"
        f" namely the clusters with or without the performance enhancing engine modification"
        f" 'TP mods'. With domain specific knowledge it is easy to understand why this is trivial"
        f" since a change in the Engine output will have a similar effect on many or"
        f" all of the other parameters thus forming a cluster however from a Airplane"
        f" Design point of view this is not such valuable information. \n\n\n"  

        f"* **5** (BR4)"
        f"  - We suspect that airplanes with a larger wing span can reach a"
        f" higher altitude, i.e. higher ceiling (Hmax): **Correct but not conclusive**." 
        f" The study on: 'Domain Specific Analysis' supports this. \n"  
        f"  - Validation: The plots with data points togethers with regression lines"
        f" for different engine types shows this trend strongly"
        f" (large positive gradient regression line) for"
        f" the piston engine type and moderately for the jet engine type (moderately positive"
        f" gradient) however it shows an opposite trend (negative gradient) for the propjet engine type. \n\n\n"

        f"* **6** (BR5)"
        f"  - We suspect that the linear relationship between the weight ratio"
        f" in the Breguet range equations and the range might in a real world"
        f" application not exhibit a non-linear relationship: **Correct**." 
        f" The study on: 'Domain Specific Analysis' supports this. \n"  
        f"  - Validation: The plot of data points togethers with a linear and a"
        f" non-linear regression line reveal that the data is only *close* to, yet not completely linear! \n\n\n"  

        f"**Please note:** Business Requirements 1 has no Hypothesis attached to it thus Hypothesis 1 is related to Business Requirements 2 and so on."
    )