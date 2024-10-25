import streamlit as st
from src.data_management import load_airplane_data, load_pkl_file

def page_airplane_feature_explanation_body():

    st.write("### Explanation of Features")

    # load data
    df = load_airplane_data()

    # text based on README file - "Dataset Content" section
    st.info(
        f"**Features**\n"

        f" **Engine Type** (categorical value) Piston, Propjet or Jet type of engine"
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Multi Engine** (categorical value) Single or multiple engines"
        f" "
        f" "
        f" "
        f" \n"

        f" **TP mods** (categorical value) Most likely: Thrust Performance modifications "
        f" "
        f" "
        f" "
        f" \n"
        
        f" **THR** Thrust for ISA (International Standard Atmosphere)"
        f" "
        f" "
        f" "
        f" \n"
        
        f" **SHP** Shaft Horse Power for ISA (International Standard Atmosphere)"
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Length** "
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Height** "
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Wing Span** "
        f" "
        f" "
        f" "
        f" \n"
        
        f" **FW** "
        f" "
        f" "
        f" "
        f" \n"
        
        f" **MEW** "
        f" "
        f" "
        f" "
        f" \n"
        
        f" **AUW** "
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Vmax** "
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Vcruise** "
        f" "
        f" "
        f" "
        f" \n"

        f" **Vstall** "
        f" "
        f" "
        f" "
        f" \n"        
        
        f" **Hmax** "
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Hmax (One)** "
        f" "
        f" "
        f" "
        f" \n"
        
        f" **ROC** "
        f" "
        f" "
        f" "
        f" \n"
        
        f" **ROC (One)** "
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Vlo** "
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Slo** "
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Vl** "
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Sl** "
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Range** "
        f" "
        f" "
        f" "
        f" \n"
        

        )

    # inspect data
    if st.checkbox("Inspect Airplane data"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df.head(10))