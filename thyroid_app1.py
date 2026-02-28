import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Thyroid Cancer Risk Prediction",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Thyroid Cancer Risk Prediction App")
st.write("Machine Learning models to predict Thyroid Cancer Risk")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload thyroid_cancer_risk_data.csv",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Drop Patient_ID if exists
    if "Patient_ID" in df.columns:
        df = df.drop(columns=["Patient_ID"])

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Encoding categorical columns
    # -------------------------------
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # -------------------------------
    # Feature / Target Split
    # -------------------------------
    X = df.drop(columns=["Diagnosis"])
    y = df["Diagnosis"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------
    # Model Selection
    # -------------------------------
    model_option = st.selectbox(
        "Select Model",
        ("Logistic Regression", "Decision Tree", "Gradient Boosting")
    )

    if model_option == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    elif model_option == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=4)

    else:
        model = GradientBoostingClassifier()

    # -------------------------------
    # Train Model
    # -------------------------------
    if st.button("Train Model"):

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)

        st.success(f"Accuracy: {accuracy * 100:.2f}%")

        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        # Feature Importance (if available)
        if hasattr(model, "feature_importances_"):
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(importance_df.set_index("Feature"))

else:
    st.info("Please upload the dataset to begin.")