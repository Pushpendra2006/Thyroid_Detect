import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Thyroid Cancer Risk Prediction",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Thyroid Cancer Risk Prediction App")
st.write("Enter patient details to predict Thyroid Cancer Risk")

# -------------------------------
# Sample Dataset (Replace with your actual dataset file path)
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("thyroid_cancer_risk_data.csv")  # Put your dataset in same folder
    if "Patient_ID" in df.columns:
        df = df.drop(columns=["Patient_ID"])
    return df

df = load_data()

# -------------------------------
# Encode Categorical Columns
# -------------------------------
le_dict = {}
categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

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

# Train model automatically
model.fit(x_train, y_train)

# -------------------------------
# Manual User Input
# -------------------------------
st.subheader("Enter Patient Details")

input_data = {}

for col in X.columns:
    if col in le_dict:
        options = le_dict[col].classes_
        selected = st.selectbox(f"{col}", options)
        input_data[col] = le_dict[col].transform([selected])[0]
    else:
        input_data[col] = st.number_input(f"{col}", value=0.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Risk"):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠ High Risk of Thyroid Cancer ({probability[1]*100:.2f}% probability)")
    else:
        st.success(f"✅ Low Risk of Thyroid Cancer ({probability[0]*100:.2f}% probability)")
