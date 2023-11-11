import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import warnings


st.set_page_config(
    page_title="Smart Customer Retention: Boosting Profits with Active Learning and LLM",
    page_icon="😶‍🌫️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def load_model(modelfile):
    loaded_model = pickle.load(open(modelfile, "rb"))

    return loaded_model


def preprocessing(df):
    df = df.copy()
    df.loc[df["Games Product"] != "No internet service", "internet_service"] = 1
    df["internet_service"].fillna(0, inplace=True)
    df["Location"] = df["Location"].map({"Jakarta": 1, "Bandung": 0})
    df["Device Class"] = df["Device Class"].map(
        {"High End": 2, "Mid End": 1, "Low End": 0}
    )
    df["Games Product"] = df["Games Product"].map(
        {"Yes": 1, "No": 0, "No internet service": 0}
    )
    df["Music Product"] = df["Music Product"].map(
        {"Yes": 1, "No": 0, "No internet service": 0}
    )
    df["Education Product"] = df["Education Product"].map(
        {"Yes": 1, "No": 0, "No internet service": 0}
    )
    df["Call Center"] = df["Call Center"].map({"Yes": 1, "No": 0})
    df["Video Product"] = df["Video Product"].map(
        {"Yes": 1, "No": 0, "No internet service": 0}
    )
    df["Use MyApp"] = df["Use MyApp"].map({"Yes": 1, "No": 0, "No internet service": 0})
    for pm in ["Digital Wallet", "Pulsa", "Debit", "Credit"]:
        df["pm_" + pm] = df["Payment Method"].map({pm: 1})
    df.drop(["Payment Method"], axis=1, inplace=True)
    df.fillna(0, inplace=True)
    return df


def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:#ff0054;text-align:left;"> Smart Customer Retention  🚀 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Predict", "Analysis"])

    with tab1:
        col1, col2 = st.columns([2, 2])
        with col1:
                st.subheader(" Find out if your customer will churn or not 🫥")
                # Categorical columns unique values
                location_values = ["Jakarta", "Bandung"]
                device_class_values = ["Low End", "Mid End", "High End"]
                games_product_values = ["Yes", "No", "No internet service"]
                music_product_values = ["Yes", "No", "No internet service"]
                education_product_values = ["Yes", "No", "No internet service"]
                call_center_values = ["Yes", "No"]
                video_product_values = ["Yes", "No", "No internet service"]
                use_myapp_values = ["Yes", "No", "No internet service"]
                payment_method_values = ["Digital Wallet", "Pulsa", "Debit", "Credit"]

                # Create input fields for categorical columns
                location = st.selectbox("Location", location_values)
                device_class = st.selectbox("Device Class", device_class_values)
                games_product = st.selectbox("Games Product", games_product_values)
                music_product = st.selectbox("Music Product", music_product_values)
                education_product = st.selectbox("Education Product", education_product_values)
                call_center = st.selectbox("Call Center", call_center_values)
                video_product = st.selectbox("Video Product", video_product_values)
                use_myapp = st.selectbox("Use MyApp", use_myapp_values)
                payment_method = st.selectbox("Payment Method", payment_method_values)

                # Create input fields for numerical columns
                # 0 to inf
                tenure_months = st.number_input("Tenure Months", 0)
                monthly_purchase = st.number_input("Monthly Purchase (Thou. IDR)", 0)
                longitude = st.number_input("Longitude", 0.0, 100000.0)
                latitude = st.number_input("Latitude", 0.0, 100000.0)
                cltv = st.number_input("CLTV (Predicted Thou. IDR)", 0)

                # Combine input values into a feature list
                feature_list = [
                    tenure_months,
                    monthly_purchase,
                    longitude,
                    latitude,
                    cltv,
                    location,
                    device_class,
                    games_product,
                    music_product,
                    education_product,
                    call_center,
                    video_product,
                    use_myapp,
                    payment_method,
                ]
                # single_pred = np.array(feature_list).reshape(1,-1)
                feat_df = pd.DataFrame(
                    [feature_list],
                    columns=[
                        "Tenure Months",
                        "Monthly Purchase (Thou. IDR)",
                        "Longitude",
                        "Latitude",
                        "CLTV (Predicted Thou. IDR)",
                        "Location",
                        "Device Class",
                        "Games Product",
                        "Music Product",
                        "Education Product",
                        "Call Center",
                        "Video Product",
                        "Use MyApp",
                        "Payment Method",
                    ],
                )
                single_pred = preprocessing(feat_df)
                single_pred = single_pred[
                    [
                        "Tenure Months",
                        "Location",
                        "Device Class",
                        "Games Product",
                        "Music Product",
                        "Education Product",
                        "Call Center",
                        "Video Product",
                        "Use MyApp",
                        "Monthly Purchase (Thou. IDR)",
                        "Longitude",
                        "Latitude",
                        "CLTV (Predicted Thou. IDR)",
                        "internet_service",
                        "pm_Credit",
                        "pm_Debit",
                        "pm_Digital Wallet",
                        "pm_Pulsa",
                    ]
                ]

                predict_button = st.button("Predict")


        with col2:
            with st.expander(" ℹ️ Information", expanded=True):
                st.write(
                    """
                Lorem Ipsum

                """
                )
            """
            ## How does it work ❓ 
            Complete all the parameters and the machine learning model will predict the probability that the customer will churn.
            """
            
            if predict_button:
                loaded_model = load_model("models/baseline_model.pkl")
                prediction = loaded_model.predict_proba(single_pred)
                col2.write(
                    """
                ## Results 🔍 
                """
                )
                col2.success(f"The probability that the customer will churn is {round(prediction[0, 1] * 100, 2)}%")

    with tab2:
        st.subheader("Analysis")
        st.write(
            """
            Lorem Ipsum
            """
        )

    # code for html ☘️ 🌾 🌳 👨‍🌾  🍃

    st.warning(
        "Note: Lorem ipsum [here](https://github.com/edutjie/dsw2023)"
    )
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
