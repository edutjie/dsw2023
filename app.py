import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import warnings
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.figure_factory as ff
import openai
from dotenv import load_dotenv

# set sns background to black
# sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black'})
plt.rcParams.update(
    {
        "lines.color": "white",
        "patch.edgecolor": "white",
        "text.color": "white",
        "axes.facecolor": "#0E1117",
        "axes.edgecolor": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": "grey",
        "figure.facecolor": "#0E1117",
        "figure.edgecolor": "#0E1117",
        "savefig.facecolor": "#0E1117",
        "savefig.edgecolor": "#0E1117",
    }
)


load_dotenv()

# openai.api_key = st.secrets["OPENAI_APIKEY"]
gpt_client = openai.OpenAI()


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
    <h1 style="color:#ff0054;text-align:left;"> Strategic Customer Retention  🚀 </h1>
    </div>
    <p>Discovering the Potency of Active Learning on Imbalanced Data, Fused with Personalized LLM-Driven Churn Solutions</p>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    df = pd.read_excel("data/Telco_customer_churn_adapted_v2.xlsx", sheet_name="Sheet1")
    df["Churn Encoded"] = df["Churn Label"].map({"Yes": 1, "No": 0})

    tab1, tab2 = st.tabs(["Predict", "Analysis"])

    with tab1:
        col1, col2 = st.columns([2, 2])
        with col1:
            st.subheader(" Find out if your customer will churn or not 🫥")
            # Categorical columns unique values
            device_class_values = ["Low End", "Mid End", "High End"]
            games_product_values = ["Yes", "No"]
            music_product_values = ["Yes", "No"]
            education_product_values = ["Yes", "No"]
            call_center_values = ["Yes", "No"]
            video_product_values = ["Yes", "No"]
            use_myapp_values = ["Yes", "No"]
            payment_method_values = ["Digital Wallet", "Pulsa", "Debit", "Credit"]

            # Create input fields for categorical columns
            device_class = st.selectbox("Device Class", device_class_values)
            call_center = st.selectbox("Use Call Center?", call_center_values)

            internet_service = st.selectbox("Has Internet Service?", ["Yes", "No"])
            if internet_service == "Yes":
                games_product = st.selectbox("Use Games Product?", games_product_values)
                music_product = st.selectbox("Use Music Product?", music_product_values)
                education_product = st.selectbox(
                    "Education Product", education_product_values
                )
                video_product = st.selectbox("Use Video Product?", video_product_values)
                use_myapp = st.selectbox("Use MyApp?", use_myapp_values)

            # if any of the games_product, music_product, education_product, video_product, use_myapp is "No internet service", then set all of them to "No internet service"
            if internet_service == "No":
                games_product = "No internet service"
                music_product = "No internet service"
                education_product = "No internet service"
                video_product = "No internet service"
                use_myapp = "No internet service"

            payment_method = st.selectbox("Payment Method", payment_method_values)
            tenure_months = st.number_input("Tenure Months", 0)
            monthly_purchase = st.number_input("Monthly Purchase (Thou. IDR)", 0)
            cltv = st.number_input("CLTV (Predicted Thou. IDR)", 0.0)

            # Combine input values into a feature list
            feature_list = [
                tenure_months,
                monthly_purchase,
                cltv,
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
                    "CLTV (Predicted Thou. IDR)",
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
                    "Device Class",
                    "Games Product",
                    "Music Product",
                    "Education Product",
                    "Call Center",
                    "Video Product",
                    "Use MyApp",
                    "Monthly Purchase (Thou. IDR)",
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
                Customers are a crucial aspect of any telecommunication business. This is a system that can offer solutions to prevent customers from leaving the company.

                The model utilized is CatBoost, which has been trained using a active learning downsampling technique.
                """
                )
            """
            ## How does it work ❓ 
            Complete all the parameters and the machine learning model will predict the probability that the customer will churn and generate the possible reason and solution.
            """

            loaded_model = load_model("models/best_model.pkl")
            if predict_button:
                with st.spinner("Predicting..."):
                    prediction = loaded_model.predict_proba(single_pred)
                    col2.write(
                        """
                    ## Results 🔍 
                    ### Churn Probability ⚖️
                    """
                    )
                    col2.success(
                        f"""
                        The probability that the customer will churn is {round(prediction[0, 1] * 100, 2)}%
                        """
                    )

                col2.write(
                    """
                    ### Reason and Solution 💡
                    """
                )
                if prediction[0, 1] >= 0.5:
                    with st.spinner("Generating Reason and Solution..."):
                        response = gpt_client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {
                                    "role": "user",
                                    "content": f"""
                                        Say you are a solution analyst. A telecommunication company wants to analyze its customer's behavior. Given a dataset that shows the usage of telecommunication services in Indonesia. 

                                        There are several columns provided in the dataset:
                                        - Customer ID (A unique customer identifier)
                                        - Tenure Months (How long the customer has been with the company by the end of the quarter specified above)
                                        - Device Class (Device classification)
                                        - Games Product (Whether the customer uses the internet service for games product)
                                        - Music Product (Whether the customer uses the internet service for music product)
                                        - Education Product (Whether the customer uses the internet service for education product)
                                        - Call Center (Whether the customer uses the call center service)
                                        - Video Product (Whether the customer uses video product service)
                                        - Use MyApp (Whether the customer uses MyApp service)
                                        - Payment Method (The method used for paying the bill)
                                        - Monthly Purchase (Total customer's monthly spent for all services with the unit of thousands of IDR)
                                        - CLTV (Customer Lifetime Value with the unit of thousands of IDR - Calculated using company's formulas)

                                        These are the unique value for all categorical columns:
                                        - Device Class unique values: ['Mid End' 'High End' 'Low End']
                                        - Games Product unique values: ['Yes' 'No' 'No internet service']
                                        - Music Product unique values: ['Yes' 'No' 'No internet service']
                                        - Education Product unique values: ['No' 'Yes' 'No internet service']
                                        - Call Center unique values: ['No' 'Yes']
                                        - Video Product unique values: ['No' 'Yes' 'No internet service']
                                        - Use MyApp unique values: ['No' 'Yes' 'No internet service']
                                        - Payment Method unique values: ['Digital Wallet' 'Pulsa' 'Debit' 'Credit']

                                        Here is the row of data (in csv format):
                                        {feat_df.to_csv(index=False)}

                                        From the given data, predict the possible reason that the customer left the company in this quarter and provide the solution to prevent the customer to leave the company!

                                        Make it in this format:
                                        Reason:
                                        ...

                                        Solution:
                                        ...
                                    """,
                                },
                            ],
                        )
                        col2.success(response.choices[0].message.content)
                else:
                    col2.success(
                        """
                        The customer is not likely to churn. No reason and solution is needed.
                        """
                    )

    with tab2:
        st.subheader("Analyze Churn 📊")
        st.write(
            """
            In this section, you can analyze the churn data vs other features and visualize it.
            """
        )
        cat_cols = df.drop(["Customer ID", "Churn Label", "Location"], axis=1).select_dtypes(include=['object']).columns.to_list()
        num_cols = df.drop(["Customer ID", "Longitude", "Latitude", "Churn Encoded"], axis=1).select_dtypes(include=['number']).columns.to_list()
        tab1, tab2 = st.tabs(["Categorical", "Numerical"])
        with tab1:
            plot_option = st.selectbox(
                label="What column do you want to analyze?",
                options=cat_cols,
                index=None,
                placeholder="Pick a column!",
            )

            if plot_option:
                st.write(
                    f"""
                    {plot_option} Distribution
                    """
                )
                data_distribution = df[plot_option].value_counts()
                st.bar_chart(data_distribution, color="#4895ef")
                st.write(
                    f"""
                    {plot_option} vs Churn Mean
                    """
                )
                data_vs_churn_mean = df.groupby(plot_option)["Churn Label"].value_counts(normalize=True).unstack()
                st.bar_chart(data_vs_churn_mean, color=["#f72585", "#4cc9f0"])
                st.write(
                    f"""
                    Insights 💡
                    """
                )
                if st.button(f"Analyze Insights for {plot_option}"):
                    with st.spinner("Generating Insights.."):
                        response = gpt_client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {
                                    "role": "user",
                                    "content": f"""
                                        Analyze these data:

                                        # {plot_option} Distribution
                                        {data_distribution.to_csv()}

                                        # {plot_option} vs Churn Mean
                                        {data_vs_churn_mean.to_csv()}

                                        Give me the insights for telecommunication company from the data in this format in points and maximum 5 points:
                                        Insights:
                                        1. ...
                                        2. ...
                                        ...
                                    """,
                                },
                            ],
                        )
                        st.success(response.choices[0].message.content)
        with tab2:
            plot_option = st.selectbox(
                label="What column do you want to analyze?",
                options=num_cols,
                index=None,
                placeholder="Pick a column!",
            )

            if plot_option:
                st.write(
                    f"""
                    {plot_option} Distribution
                    """
                )
                # make a slider
                bin_size = st.slider("Bin Size", min_value=1, max_value=int(df[plot_option].max())//2, value=5)
                st.plotly_chart(
                    ff.create_distplot(
                        [df[plot_option].to_list()],
                        group_labels=[plot_option],
                        colors=["#4895ef"],
                        bin_size=bin_size,
                    ),
                    use_container_width=True,
                )

                # data mean vs churn
                data_mean_vs_churn = df.groupby("Churn Label")[plot_option].mean()
                st.write(
                    f"""
                    {plot_option} Mean vs Churn
                    > Mean of {plot_option} (Churn=No): {data_mean_vs_churn[0]}
                    >
                    > Mean of {plot_option} (Churn=Yes): {data_mean_vs_churn[1]}
                    """
                )
                st.pyplot(sns.barplot(x=data_mean_vs_churn.index, y=data_mean_vs_churn.values, palette=["#f72585", "#4cc9f0"]).figure)
                # st.bar_chart(data_mean_vs_churn, color=["#f72585", "#4cc9f0])
                st.write(
                    f"""
                    Insights 💡
                    """
                )
                if st.button(f"Analyze Insights for {plot_option}"):
                    with st.spinner("Generating Insights.."):
                        response = gpt_client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {
                                    "role": "user",
                                    "content": f"""
                                        Analyze these data:

                                        # {plot_option} Mean vs Churn
                                        {data_mean_vs_churn.to_csv()}

                                        Give me the insights for telecommunication company from the data in this format in points and maximum 5 points:
                                        Insights:
                                        1. ...
                                        2. ...
                                        ...
                                    """,
                                },
                            ],
                        )
                        st.success(response.choices[0].message.content)
                


        # with tab2:
            

        st.subheader("Location Visualization")
        col1, col2 = st.columns([2, 2])
        with col1:
            with open("maps/map.html", "r") as html_file:
                folium_html_content = html_file.read()
            st.components.v1.html(folium_html_content, height=400)

        with col2:
            with open("maps/geo_map.html", "r") as html_file:
                folium_html_content2 = html_file.read()
            st.components.v1.html(folium_html_content2, height=400)

        # st_folium(m, width=725)

    # code for html ☘️ 🌾 🌳 👨‍🌾  🍃

    st.warning("You can access the repository [here](https://github.com/edutjie/dsw2023).")
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
