import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Streamlit page settings
# -----------------------------
st.set_page_config(
    page_title="Car Price Prediction App",
    page_icon="🚗",
    layout="centered"
)


# -----------------------------
# Helper functions
# -----------------------------
def find_dataset_file():
    """Find the dataset file inside the project folder."""
    possible_files = [
        "hatla2ee_cars_august_2025_clean.xlsx",
        "hatla2ee_cars_august_2025_clean.csv",
        "cars.csv",
        "cars.xlsx",
        "dataset.csv",
        "dataset.xlsx",
    ]

    for file_name in possible_files:
        try:
            if file_name.endswith(".xlsx"):
                pd.read_excel(file_name, nrows=1)
            else:
                pd.read_csv(file_name, nrows=1)
            return file_name
        except FileNotFoundError:
            continue
        except Exception:
            return file_name

    return None


def load_dataset(file_name):
    """Load CSV or Excel dataset."""
    if file_name.endswith(".xlsx"):
        return pd.read_excel(file_name)
    return pd.read_csv(file_name)


def clean_price_column(cars):
    """Create numeric price column."""
    if "price in EGP" in cars.columns:
        price_source = cars["price in EGP"]
    elif "price" in cars.columns:
        price_source = cars["price"]
    elif "Price" in cars.columns:
        price_source = cars["Price"]
    else:
        raise ValueError("No price column found. Expected 'price', 'Price', or 'price in EGP'.")

    cars["price in EGP"] = (
        price_source.astype(str)
        .str.replace("EGP", "", regex=False)
        .str.replace("egp", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    cars["price in EGP"] = pd.to_numeric(cars["price in EGP"], errors="coerce")
    return cars


def clean_mileage_column(cars):
    """Create numeric mileage column."""
    if "mileage in KM" in cars.columns:
        mileage_source = cars["mileage in KM"]
    elif "mileage" in cars.columns:
        mileage_source = cars["mileage"]
    elif "Mileage" in cars.columns:
        mileage_source = cars["Mileage"]
    else:
        raise ValueError("No mileage column found. Expected 'mileage', 'Mileage', or 'mileage in KM'.")

    cars["mileage in KM"] = (
        mileage_source.astype(str)
        .str.replace("Km", "", regex=False)
        .str.replace("KM", "", regex=False)
        .str.replace("km", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    cars["mileage in KM"] = pd.to_numeric(cars["mileage in KM"], errors="coerce")
    return cars


def clean_year_column(cars):
    """Create numeric year column."""
    if "year" in cars.columns:
        year_source = cars["year"]
    elif "Year" in cars.columns:
        year_source = cars["Year"]
    else:
        raise ValueError("No year column found. Expected 'year' or 'Year'.")

    cars["year"] = pd.to_numeric(year_source, errors="coerce")
    return cars


def get_company_column(cars):
    """Normalize company column name."""
    if "company" in cars.columns:
        cars["company"] = cars["company"].astype(str).str.strip()
    elif "Company" in cars.columns:
        cars["company"] = cars["Company"].astype(str).str.strip()
    else:
        raise ValueError("No company column found. Expected 'company' or 'Company'.")
    return cars


@st.cache_resource
def train_models_inside_app():
    """Load the dataset, clean it, build features, and train all models inside Streamlit."""
    dataset_file = find_dataset_file()

    if dataset_file is None:
        raise FileNotFoundError(
            "Dataset file was not found. Please upload the dataset beside app.py. "
            "Expected name: hatla2ee_cars_august_2025_clean.xlsx"
        )

    cars = load_dataset(dataset_file)

    cars = get_company_column(cars)
    cars = clean_year_column(cars)
    cars = clean_mileage_column(cars)
    cars = clean_price_column(cars)

    # Keep only the columns needed for the Streamlit prediction app
    cars = cars.dropna(subset=["company", "year", "mileage in KM", "price in EGP"])

    # Remove unrealistic values
    cars = cars[
        (cars["year"] >= 1970) &
        (cars["year"] <= 2026) &
        (cars["mileage in KM"] >= 0) &
        (cars["mileage in KM"] <= 1_000_000) &
        (cars["price in EGP"] > 0)
    ].copy()

    # Remove price outliers using IQR method
    q1 = cars["price in EGP"].quantile(0.25)
    q3 = cars["price in EGP"].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    cars = cars[cars["price in EGP"] <= upper_bound].copy()

    company_list = sorted(cars["company"].dropna().unique().tolist())
    numeric_columns = ["year", "mileage in KM"]

    # One-Hot Encoding for company
    company_dummies = pd.get_dummies(
        cars[["company"]],
        prefix=["company"],
        dtype=int,
        drop_first=True
    )

    numeric_features = cars[numeric_columns].copy()
    x = pd.concat([company_dummies, numeric_features], axis=1)
    y_regression = cars["price in EGP"]

    # Scale the numeric columns for regression and classification
    regression_scaler = StandardScaler()
    x_scaled = x.copy()
    x_scaled[numeric_columns] = regression_scaler.fit_transform(x_scaled[numeric_columns])

    # Regression model
    regression_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    regression_model.fit(x_scaled, y_regression)

    # Classification target based on price quantiles
    low_limit = y_regression.quantile(0.33)
    high_limit = y_regression.quantile(0.66)

    y_classification = pd.cut(
        y_regression,
        bins=[-np.inf, low_limit, high_limit, np.inf],
        labels=["Low Price", "Medium Price", "High Price"]
    )

    classification_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    classification_model.fit(x_scaled, y_classification)

    # Clustering features include X features + price
    cluster_features = x.copy()
    cluster_features["price in EGP"] = y_regression.values

    cluster_scaler = StandardScaler()
    cluster_features_scaled = cluster_scaler.fit_transform(cluster_features)

    clustering_model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )
    clustering_model.fit(cluster_features_scaled)

    return {
        "regression_model": regression_model,
        "classification_model": classification_model,
        "clustering_model": clustering_model,
        "regression_scaler": regression_scaler,
        "cluster_scaler": cluster_scaler,
        "feature_columns": list(x.columns),
        "cluster_feature_columns": list(cluster_features.columns),
        "numeric_columns": numeric_columns,
        "company_list": company_list,
        "dataset_file": dataset_file,
        "training_rows": len(cars)
    }


# -----------------------------
# Train models
# -----------------------------
try:
    objects = train_models_inside_app()
except Exception as error:
    st.error("An error occurred while loading the dataset or training the models.")
    st.exception(error)
    st.stop()

regression_model = objects["regression_model"]
classification_model = objects["classification_model"]
clustering_model = objects["clustering_model"]
regression_scaler = objects["regression_scaler"]
cluster_scaler = objects["cluster_scaler"]
feature_columns = objects["feature_columns"]
cluster_feature_columns = objects["cluster_feature_columns"]
numeric_columns = objects["numeric_columns"]
company_list = objects["company_list"]


# -----------------------------
# App interface
# -----------------------------
st.title("🚗 Car Price Prediction, Classification and Clustering")

st.write("""
This application trains the machine learning models directly inside Streamlit,
then predicts the car price, classifies the price category, and assigns the car to a cluster.
""")

st.info(
    f"Dataset used: {objects['dataset_file']} | Training rows after cleaning: {objects['training_rows']:,}"
)


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
    if "year" in input_data.columns:
        input_data.loc[0, "year"] = year

    if "mileage in KM" in input_data.columns:
        input_data.loc[0, "mileage in KM"] = mileage

    # Add company dummy column if it exists
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
    regression_input[numeric_columns] = regression_scaler.transform(
        regression_input[numeric_columns]
    )

    predicted_price = regression_model.predict(regression_input)[0]

    # Classification prediction
    classification_input = regression_input.copy()
    predicted_category = classification_model.predict(classification_input)[0]

    # Clustering prediction
    cluster_input = raw_input.copy()
    cluster_input["price in EGP"] = predicted_price

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
