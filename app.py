import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load saved models and objects
# -----------------------------
regression_model = joblib.load('streamlit_regression_model.pkl')
regression_scaler = joblib.load('streamlit_regression_scaler.pkl')

classification_model = joblib.load('streamlit_classification_model.pkl')
classification_scaler = joblib.load('streamlit_classification_scaler.pkl')

clustering_model = joblib.load('streamlit_clustering_model.pkl')
cluster_scaler = joblib.load('streamlit_cluster_scaler.pkl')

feature_info = joblib.load('streamlit_feature_info.pkl')
feature_columns = feature_info['feature_columns']
cluster_feature_columns = feature_info['cluster_feature_columns']
numeric_columns = feature_info['numeric_columns']
company_list = feature_info['company_list']

# -----------------------------
# Streamlit page settings
# -----------------------------
st.set_page_config(
    page_title='Car Price Prediction App',
    page_icon='🚗',
    layout='centered'
)

st.title('🚗 Car Price Prediction, Classification and Clustering')

st.write("""
This application predicts the car price, classifies the price category,
and assigns the car to a cluster based on the trained machine learning models.
""")

# -----------------------------
# User inputs
# -----------------------------
st.sidebar.header('Enter Car Information')

company = st.sidebar.selectbox(
    'Car Company',
    company_list
)

year = st.sidebar.number_input(
    'Manufacturing Year',
    min_value=1980,
    max_value=2030,
    value=2020,
    step=1
)

mileage = st.sidebar.number_input(
    'Mileage in KM',
    min_value=0,
    max_value=1000000,
    value=50000,
    step=1000
)

# -----------------------------
# Function to prepare features
# -----------------------------
def prepare_input(company, year, mileage):
    # Create empty row with the same columns used in training
    input_data = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns
    )

    # Add numeric values
    if 'year' in input_data.columns:
        input_data.loc[0, 'year'] = year

    if 'mileage in KM' in input_data.columns:
        input_data.loc[0, 'mileage in KM'] = mileage

    # Add company dummy column if it exists
    company_column = 'company_' + str(company)
    if company_column in input_data.columns:
        input_data.loc[0, company_column] = 1

    return input_data

# -----------------------------
# Prediction button
# -----------------------------
if st.button('Predict'):

    raw_input = prepare_input(company, year, mileage)

    # Regression prediction
    regression_input = raw_input.copy()
    regression_input[numeric_columns] = regression_scaler.transform(
        regression_input[numeric_columns]
    )

    predicted_price = regression_model.predict(regression_input)[0]

    # Classification prediction
    classification_input = raw_input.copy()
    classification_input_scaled = classification_scaler.transform(classification_input)

    predicted_category = classification_model.predict(classification_input_scaled)[0]

    # Clustering prediction
    cluster_input = raw_input.copy()
    cluster_input['Price'] = predicted_price

    # Make sure cluster input has the same columns and order used in training
    cluster_input = cluster_input.reindex(
        columns=cluster_feature_columns,
        fill_value=0
    )

    cluster_input_scaled = cluster_scaler.transform(cluster_input)
    predicted_cluster = clustering_model.predict(cluster_input_scaled)[0]

    # -----------------------------
    # Display results
    # -----------------------------
    st.subheader('Prediction Results')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label='Predicted Price',
            value=f'{predicted_price:,.0f} EGP'
        )

    with col2:
        st.metric(
            label='Price Category',
            value=str(predicted_category)
        )

    with col3:
        st.metric(
            label='Cluster',
            value=int(predicted_cluster)
        )

    st.write('---')
    st.write('### Input Summary')

    st.dataframe(pd.DataFrame({
        'Company': [company],
        'Year': [year],
        'Mileage in KM': [mileage],
        'Predicted Price': [round(predicted_price, 2)],
        'Price Category': [predicted_category],
        'Cluster': [int(predicted_cluster)]
    }))
