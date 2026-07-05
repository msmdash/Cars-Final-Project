import os
import glob

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------
# Streamlit page settings
# -----------------------------
st.set_page_config(
    page_title="Car Price Prediction App",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Price Prediction, Classification and Clustering")

st.write("""
This application trains the machine learning models directly from the dataset,
then predicts the car price, classifies the price category, and assigns the car to a cluster.
""")

# -----------------------------
# Helper functions
# -----------------------------
def clean_price(value):
    value = str(value)
    value = value.replace("EGP", "")
    value = value.replace(",", "")
    value = value.strip()
    return pd.to_numeric(value, errors="coerce")


def clean_mileage(value):
    value = str(value)
    value = value.replace("Km", "")
    value = value.replace("KM", "")
    value = value.replace("km", "")
    value = value.replace(",", "")
    value = value.strip()
    return pd.to_numeric(value, errors="coerce")


def find_dataset_file():
    """Find a dataset file in the same folder as app.py."""
    preferred_files = [
        "hatla2ee_cars_august_2025_clean.xlsx",
        "hatla2ee_cars_august_2025.xlsx",
        "hatla2ee_cars_august_2025_clean.csv",
        "hatla2ee_cars_august_2025.csv",
        "cars.csv",
        "cars.xlsx"
    ]

    for file_name in preferred_files:
        if os.path.exists(file_name):
            return file_name

    # Fallback: choose the first csv/xlsx file in the repository folder
    data_files = []
    data_files.extend(glob.glob("*.csv"))
    data_files.extend(glob.glob("*.xlsx"))

    # Avoid temporary Excel files
    data_files = [file for file in data_files if not os.path.basename(file).startswith("~$")]

    if len(data_files) == 0:
        return None

    return data_files[0]


@st.cache_data
def load_dataset():
    dataset_file = find_dataset_file()

    if dataset_file is None:
        raise FileNotFoundError(
            "No dataset file was found. Please upload a CSV or Excel dataset to the GitHub repository."
        )

    if dataset_file.lower().endswith(".csv"):
        cars = pd.read_csv(dataset_file)
    else:
        cars = pd.read_excel(dataset_file)

    return cars, dataset_file


@st.cache_resource
def train_models():
    cars, dataset_file = load_dataset()

    # Normalize column names by removing extra spaces
    cars.columns = cars.columns.str.strip()

    # -----------------------------
    # Detect important columns
    # -----------------------------
    price_col = "price in EGP" if "price in EGP" in cars.columns else "price"
    mileage_col = "mileage in KM" if "mileage in KM" in cars.columns else "mileage"

    required_columns = ["company", "year", price_col, mileage_col]
    missing_columns = [col for col in required_columns if col not in cars.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in dataset: {missing_columns}. "
            f"Available columns are: {list(cars.columns)}"
        )

    # -----------------------------
    # Basic preprocessing
    # -----------------------------
    cars["price in EGP"] = cars[price_col].apply(clean_price)
    cars["mileage in KM"] = cars[mileage_col].apply(clean_mileage)
    cars["year"] = pd.to_numeric(cars["year"], errors="coerce")
    cars["company"] = cars["company"].astype(str).str.strip()

    cars = cars.dropna(subset=["company", "year", "mileage in KM", "price in EGP"])

    cars = cars[
        (cars["year"] >= 1970) &
        (cars["year"] <= 2026) &
        (cars["mileage in KM"] >= 0) &
        (cars["mileage in KM"] <= 1_000_000) &
        (cars["price in EGP"] > 0)
    ]

    # Remove price outliers using IQR
    Q1 = cars["price in EGP"].quantile(0.25)
    Q3 = cars["price in EGP"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    cars = cars[
        (cars["price in EGP"] >= lower_bound) &
        (cars["price in EGP"] <= upper_bound)
    ]

    if cars.empty:
        raise ValueError("No data remained after cleaning. Please check the dataset values.")

    company_list = sorted(cars["company"].dropna().unique().tolist())

    # -----------------------------
    # Feature engineering
    # -----------------------------
    numeric_columns = ["year", "mileage in KM"]

    company_dummies = pd.get_dummies(
        cars[["company"]],
        prefix=["company"],
        dtype=int,
        drop_first=True
    )

    x_numeric = cars[numeric_columns].reset_index(drop=True)
    x_dummies = company_dummies.reset_index(drop=True)
    x = pd.concat([x_dummies, x_numeric], axis=1)

    y_regression = cars["price in EGP"].reset_index(drop=True)

    # Classification target based on price quantiles
    low_limit = y_regression.quantile(0.33)
    high_limit = y_regression.quantile(0.66)

    y_classification = pd.cut(
        y_regression,
        bins=[-np.inf, low_limit, high_limit, np.inf],
        labels=["Low Price", "Medium Price", "High Price"]
    )

    # Scale numeric columns for consistent model input
    scaler = StandardScaler()
    x_scaled = x.copy()
    x_scaled[numeric_columns] = scaler.fit_transform(x_scaled[numeric_columns])

    # -----------------------------
    # Train models
    # -----------------------------
    regression_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    regression_model.fit(x_scaled, y_regression)

    classification_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    classification_model.fit(x_scaled, y_classification)

    # Clustering uses the same input features plus the real price during training
    cluster_features = x.copy()
    cluster_features["Price"] = y_regression.values

    cluster_scaler = StandardScaler()
    cluster_features_scaled = cluster_scaler.fit_transform(cluster_features)

    clustering_model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )
    clustering_model.fit(cluster_features_scaled)

    training_info = {
        "dataset_file": dataset_file,
        "rows_after_cleaning": len(cars),
        "feature_columns": list(x.columns),
        "cluster_feature_columns": list(cluster_features.columns),
        "numeric_columns": numeric_columns,
        "company_list": company_list,
        "price_lower_bound": lower_bound,
        "price_upper_bound": upper_bound
    }

    return regression_model, classification_model, clustering_model, scaler, cluster_scaler, training_info


try:
    with st.spinner("Training models from dataset... Please wait."):
        (
            regression_model,
            classification_model,
            clustering_model,
            scaler,
            cluster_scaler,
            training_info
        ) = train_models()

except Exception as error:
    st.error("The application could not train the models.")
    st.write("Error details:")
    st.exception(error)
    st.stop()

feature_columns = training_info["feature_columns"]
cluster_feature_columns = training_info["cluster_feature_columns"]
numeric_columns = training_info["numeric_columns"]
company_list = training_info["company_list"]

with st.expander("Training information"):
    st.write("Dataset file:", training_info["dataset_file"])
    st.write("Rows used after cleaning:", training_info["rows_after_cleaning"])
    st.write("Number of features:", len(feature_columns))

# -----------------------------
# User inputs
# -----------------------------
st.sidebar.header("Enter Car Information")

company = st.sidebar.selectbox(
    "Car Company",
    company_list
)

year = st.sidebar.number_input(
    "Manufacturing Year",
    min_value=1980,
    max_value=2030,
    value=2020,
    step=1
)

mileage = st.sidebar.number_input(
    "Mileage in KM",
    min_value=0,
    max_value=1_000_000,
    value=50_000,
    step=1_000
)

# -----------------------------
# Function to prepare features
# -----------------------------
def prepare_input(company, year, mileage):
    input_data = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns
    )

    if "year" in input_data.columns:
        input_data.loc[0, "year"] = year

    if "mileage in KM" in input_data.columns:
        input_data.loc[0, "mileage in KM"] = mileage

    company_column = "company_" + str(company)
    if company_column in input_data.columns:
        input_data.loc[0, company_column] = 1

    return input_data

# -----------------------------
# Prediction button
# -----------------------------
if st.button("Predict"):

    raw_input = prepare_input(company, year, mileage)

    # Regression prediction
    regression_input = raw_input.copy()
    regression_input[numeric_columns] = scaler.transform(
        regression_input[numeric_columns]
    )

    predicted_price = regression_model.predict(regression_input)[0]

    # Classification prediction
    predicted_category = classification_model.predict(regression_input)[0]

    # Clustering prediction
    cluster_input = raw_input.copy()
    cluster_input["Price"] = predicted_price

    cluster_input = cluster_input.reindex(
        columns=cluster_feature_columns,
        fill_value=0
    )

    cluster_input_scaled = cluster_scaler.transform(cluster_input)
    predicted_cluster = clustering_model.predict(cluster_input_scaled)[0]

    # -----------------------------
    # Display results
    # -----------------------------
    st.subheader("Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Predicted Price",
            value=f"{predicted_price:,.0f} EGP"
        )

    with col2:
        st.metric(
            label="Price Category",
            value=str(predicted_category)
        )

    with col3:
        st.metric(
            label="Cluster",
            value=int(predicted_cluster)
        )

    st.write("---")
    st.write("### Input Summary")

    st.dataframe(pd.DataFrame({
        "Company": [company],
        "Year": [year],
        "Mileage in KM": [mileage],
        "Predicted Price": [round(predicted_price, 2)],
        "Price Category": [predicted_category],
        "Cluster": [int(predicted_cluster)]
    }))
