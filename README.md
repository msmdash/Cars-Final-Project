# Car Price Prediction AI Project

This project predicts used car prices using Machine Learning and provides an interactive Streamlit web application.

## Project Features

- Predicts the expected car price using a Regression model.
- Classifies the car into a price category using a Classification model.
- Assigns the car to a cluster using a Clustering model.
- Provides a simple online user interface using Streamlit.

## Required Files

Make sure these files exist in the same repository:

- `app.py`
- `requirements.txt`
- `streamlit_regression_model.pkl`
- `streamlit_regression_scaler.pkl`
- `streamlit_classification_model.pkl`
- `streamlit_classification_scaler.pkl`
- `streamlit_clustering_model.pkl`
- `streamlit_cluster_scaler.pkl`
- `streamlit_feature_info.pkl`

## How to Run Locally

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

## How to Upload to GitHub

1. Create a new repository on GitHub.
2. Put all project files in one folder.
3. Open Terminal or Git Bash inside the project folder.
4. Run the following commands:

```bash
git init
git add .
git commit -m "Initial car price prediction Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
git push -u origin main
```

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY_NAME` with your real GitHub username and repository name.

## How to Deploy Online Using Streamlit Community Cloud

1. Go to Streamlit Community Cloud.
2. Sign in using your GitHub account.
3. Click **New app**.
4. Select your GitHub repository.
5. Set the main file path to:

```text
app.py
```

6. Click **Deploy**.

After deployment, Streamlit will give you a public link for the application.

## Notes

If deployment fails, check that:

- `requirements.txt` exists.
- `app.py` exists.
- All `.pkl` model files are uploaded.
- File names in GitHub match exactly the file names used inside `app.py`.
