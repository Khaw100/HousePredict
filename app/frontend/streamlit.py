import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import requests
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

st.set_page_config(page_title="Housing Price Prediction", layout="wide")

BAACKEND_URL = "http://backend:5000/predict" 
METRICS_URL = "http://backend:5000/metrics"


# Header Section
st.title("My Portfolio with Streamlit")
st.markdown("""
This is a simple wep portfolio app built using streamlit to showcase:
- Profile Summary
- Projects
- Housing Price Prediction Interface
- Data Visualization and Model Performance
""")

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs([
    "Profile Summary", 
    "Projects", 
    "Housing Price Prediction", 
    "Data Visualization & Model Performance"])

with tab1:
    st.header("Profile Summary")
    st.markdown("""
    - Name: Muhammad Rakha Wiratama
    - Background: Data Science Enthusiast, Software Developer
    - Skills: Python, Machine Learning, MLOps, Deep Learning, GenAI
    - Email: wiratamarakha@gmail.com 
    """)

with tab2:
    st.header("Projects")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Project 1: Housing Price Prediction")
        st.image(
            os.path.join(ASSETS_DIR, "houses.png"),
            width=200
        )
        st.markdown("""
        - Developed a machine learning model to predict housing prices using XGBoost.
        - Implemented feature engineering, data preprocessing, and model evaluation.
        - Deployed the model using Flask and MLflow for real-time predictions.
        """)

    with col2:
        st.subheader("Project 2: Fashion Recommendation System")
        st.image(
            os.path.join(ASSETS_DIR, "fashion.png"),
            width=200
        )
        st.markdown("""
        - Built a recommendation system for fashion products using collaborative filtering and content-based filtering.
        - Utilized user behavior data and product attributes to enhance recommendation accuracy.
        """)
    
    with col3:
        st.subheader("Project 3: Finetuning GPT-2 for Psychological Counseling")
        st.image(
            os.path.join(ASSETS_DIR, "gpt2.png"),
            width=200
        )
        st.markdown("""
        - Fine-tuned GPT-2 model to provide psychological counseling responses.
        - Leveraged transfer learning techniques to adapt the model for empathetic and context-aware interactions.
        """)


with tab3:
    st.header("Housing Price Prediction Interface")
    st.subheader("Input Housing Features (Upload CSV File for Prediction)")

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is None:
        st.info("Please upload a CSV file to enable prediction")
        st.stop()

    input_data = pd.read_csv(uploaded_file)
    st.write("Input Data:")
    st.dataframe(input_data)

    # Check if uploaded file is empty
    if input_data.empty:
        st.error("Uploaded CSV is empty.")
        st.stop()

    if st.button("Predict Housing Prices"):
        with st.spinner("Sending data to backend for prediction..."):
            response = pd.DataFrame()
            try:
                response = pd.read_json(
                    pd.io.json.dumps(
                        requests.post(
                            BAACKEND_URL,
                            json={"inputs": input_data.to_dict(orient="records")},
                            timeout=30
                        )
                    )
                )

                if response.status_code != 200:
                    st.error(f"Backend error: {response.text}")
                    st.stop()

                predictions = response.json().get("predictions", [])
                input_data["PredictedPrice"] = predictions

                st.success("Prediction successful!")
                st.subheader("Predicted Housing Prices:")
                st.dataframe(input_data)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

with tab4:
    st.header("Data Visualization & Model Performance")
    st.subheader("Dataset Visualization")

    if 'input_data' in locals():
        numeric_cols = input_data.select_dtypes(include="number").columns.tolist()

        if numeric_cols:
            feature = st.selectbox("Select Feature for Visualization", numeric_cols)

            # Display distribution plot
            fig, ax = plt.subplots()
            ax.hist(input_data[feature], bins=30)
            ax.set_title(f"Distribution of {feature}")
            st.pyplot(fig)

            # Displat Correlation Heatmap
            st.markdown("### Correlation Heatmap")
            corr = input_data[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.info("No numeric columns available for visualization.")
    else:
        st.info("Please upload data in the 'Housing Price Prediction' tab to enable visualization.")


    st.subheader("Model Performance Metrics")
    try:
        response = requests.get(METRICS_URL, timeout=10)
        if response.status_code == 200:
            metrics = response.json()
            training = metrics.get("training_metrics", {})
            st.markdown(f"**Mean Squared Error (MSE):** {round(training.get('mse', 0), 2)}")
            st.markdown(f"**Root Mean Squared Error (RMSE):** {round(training.get('rmse', 0), 2)}")
            st.markdown(f"**Mean Absolute Error (MAE):** {round(training.get('mae', 0), 2)}")
        else:
            st.error(f"Failed to fetch metrics: {response.text}")
    except Exception as e:
        st.warning(f"Could not retrieve metrics from backend: {e}")


    st.subheader("Prediction Monitoring")

    try:
        if response.get("recent_predictions"):
            st.dataframe(pd.DataFrame(response["recent_predictions"]))
        else:
            st.info("No prediction logs available yet.")
    except:
        pass







