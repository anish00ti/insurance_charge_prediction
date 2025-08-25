import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(page_title="Medical Charges Prediction", layout="wide")

st.title("💡 Medical Insurance Charges Prediction (Linear Regression)")
st.markdown("A simple ML app to predict medical charges using **Linear Regression**.")

# -----------------------
# Load Data
# -----------------------
data = pd.read_csv("ready.csv")
data.drop(columns=['sex', 'smoker'], inplace=True)

# -----------------------
# Train Model
# -----------------------
x = data.drop(columns=['charges'])
y = data['charges']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

# -----------------------
# Model Performance
# -----------------------
r2 = r2_score(y_test, y_predict)
n = x_test.shape[0]  # samples
p = x_test.shape[1]  # features
adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

# Display metrics
st.subheader("📊 Model Performance")
col1, col2 = st.columns(2)
col1.metric("R²", f"{r2:.3f}")
col2.metric("Adjusted R²", f"{adjusted_r2:.3f}")

# -----------------------
# Prediction vs Actual
# -----------------------
st.subheader("🔍 Prediction vs Actual Comparison")

df_results = pd.DataFrame({"Actual": y_test.values, "Predicted": y_predict})
st.dataframe(df_results.head(20))

# Plot Actual vs Predicted
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_predict, alpha=0.7, ax=ax)
sns.lineplot(x=y_test, y=y_test, color="red", label="Perfect Fit", ax=ax)
ax.set_xlabel("Actual Charges")
ax.set_ylabel("Predicted Charges")
ax.set_title("Actual vs Predicted Charges")
st.pyplot(fig)

# -----------------------
# User Input Prediction
# -----------------------
st.subheader("🧑‍⚕️ Try It Yourself")
st.markdown("Enter values to predict charges:")

col1, col2, col3 = st.columns(3)

age = col1.number_input("Age (Standardized)", value=0.0, step=0.1)
bmi = col2.number_input("BMI (Standardized)", value=0.0, step=0.1)
children = col3.number_input("Children (Standardized)", value=0.0, step=0.1)

is_smoker = st.selectbox("Smoker?", [0, 1])
is_male = st.selectbox("Male?", [0, 1])
is_northwest = st.selectbox("Region Northwest?", [0, 1])
is_southeast = st.selectbox("Region Southeast?", [0, 1])
is_southwest = st.selectbox("Region Southwest?", [0, 1])

user_data = np.array([[age, bmi, children, is_smoker, is_male, is_northwest, is_southeast, is_southwest]])
prediction = model.predict(user_data)[0]

st.success(f"💰 Predicted Insurance Charge: **${prediction:,.2f}**")
