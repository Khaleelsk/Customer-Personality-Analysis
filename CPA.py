import streamlit as st
import pandas as pd
import joblib
from datetime import date

# Load any pre-trained model or data if applicable
try:
    with open("CPA_model.pkl", "rb") as file:
        model = joblib.load(file)
except FileNotFoundError:
    model = None
    st.error("Model file not found. Please upload 'CPA_model.pkl'.")
except Exception as e:
    model = None
    st.error(f"Incompatible model file or error loading the model: {e}")

# App title
st.title("Customer Personality Analysis")

# Sidebar inputs
st.sidebar.header("Customer Details")

# Input fields for customer data
education = st.sidebar.selectbox("Education Level", ["Undergraduate","Graduate","Postgraduate", "PhD"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single","Couple"])
income = st.sidebar.number_input("Annual Income (USD)", min_value=0, step=1000)
recency = st.sidebar.slider("Days Since Last Purchase", 0, 365, 30)
complain = st.sidebar.selectbox("Customer Complained?", ["No", "Yes"])
age = st.sidebar.number_input("Age", min_value=18, max_value=100, step=1)
tot_expenses = st.sidebar.number_input("Total Expenses (USD)", min_value=0, step=100)
tot_accepted_cmp = st.sidebar.slider("Total Campaigns Accepted", 0, 10, 1)
tot_purchases = st.sidebar.number_input("Total Purchases", min_value=0, step=1)
tot_children = st.sidebar.slider("Total Children", 0, 10, 0)
tot_adults = st.sidebar.slider("Total Adults", 1, 10, 1)
customer_since = st.sidebar.date_input("Customer Since", value=date(2010, 1, 1))

# Calculate family size
family_size = tot_children + tot_adults

# Process and display insights
st.header("Customer Insights")

# Display input summary
data = {
    "Education": education,
    "Marital Status": marital_status,
    "Income (USD)": income,
    "Recency (Days)": recency,
    "Complain": complain,
    "Age": age,
    "Total Expenses (USD)": tot_expenses,
    "Total Accepted Campaigns": tot_accepted_cmp,
    "Total Purchases": tot_purchases,
    "Total Children": tot_children,
    "Total Adults": tot_adults,
    "Family Size": family_size,
    "Customer Since": customer_since,
}
data_df = pd.DataFrame([data])

st.subheader("Customer Profile")
st.dataframe(data_df)

# Analyze customer personality
st.subheader("Analysis")
if income < 20000:
    st.write("This customer may belong to a low-income segment.")
elif 20000 <= income <= 80000:
    st.write("This customer belongs to a middle-income segment.")
else:
    st.write("This customer belongs to a high-income segment.")

if tot_accepted_cmp > 5:
    st.write("The customer is highly responsive to marketing campaigns.")
else:
    st.write("The customer has low responsiveness to marketing campaigns.")

if tot_purchases / (family_size or 1) > 10:
    st.write("This customer is a frequent purchaser.")
else:
    st.write("This customer has average or low purchase frequency.")

if complain == "Yes":
    st.write("The customer has filed complaints previously, indicating potential dissatisfaction.")
else:
    st.write("The customer has not filed complaints, indicating potential satisfaction.")

st.write("**Customer has been with us since:**", customer_since.strftime("%B %d, %Y"))

# Predict personality if model is available
if model:
    try:
        input_data = pd.DataFrame([{
            "Education": education,
            "Marital Status": marital_status,
            "Income": income,
            "Recency": recency,
            "Complain": 1 if complain == "Yes" else 0,
            "Age": age,
            "Total Expenses": tot_expenses,
            "Total Accepted Campaigns": tot_accepted_cmp,
            "Total Purchases": tot_purchases,
            "Family Size": family_size,
            "Customer Since": (date.today() - customer_since).days,
        }])
        prediction = model.predict(input_data)
        st.subheader("Predicted Personality")
        st.write("Predicted Personality Type:", prediction[0])
    except Exception as e:
        st.error(f"Error in prediction: {e}")
else:
    st.info("No pre-trained model found or an issue with the loaded model. Upload a valid 'CPA_model.pkl' file to enable personality prediction.")
