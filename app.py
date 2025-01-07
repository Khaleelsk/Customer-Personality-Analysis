import streamlit as st
import pandas as pd
import pickle

# Preprocessing function
def preprocess_data(data):
    # Handle missing values if needed
    if 'Income' in data.columns:
        data['Income'].fillna(data['Income'].median(), inplace=True)

    # Convert categorical columns to numeric if necessary
    if 'Education' in data.columns:
        education_map = {"Undergraduate": 1, "Graduation": 2, "Postgraduate": 3, "PhD": 4}
        data['Education'] = data['Education'].map(education_map)

    if 'Marital_Status' in data.columns:
        marital_status_map = {"Single": 0, "Couple": 1}
        data['Marital_Status'] = data['Marital_Status'].map(marital_status_map)

    # Drop any extra columns to ensure model compatibility
    expected_features = ["Education", "Marital_Status", "Income", "Recency", "Complain", "Response", "Age", "Tot_Expenses", "Tot_AcceptedCmp", "Tot_Purchases", "Tot_Children", "Tot_adults", "Customer_Since"]
    data = data[expected_features]

    return data

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    with open("CPA_model.pkl", "rb") as file:
        return pickle.load(file)

# Streamlit app
st.title("Customer Personality Analysis & Segmentation")

st.write("""
This app analyzes customer data to:
- Segment customers based on personality traits, demographics, and purchasing patterns.
- Provide tailored recommendations for marketing and customer engagement.
- Predict customer behaviors and improve overall experience.
""")

# Load model
model = load_model()

# User input form
st.header("Enter Customer Details")

# User input fields
education = st.selectbox("Education", ["Undergraduate","Graduation", "Postgraduate", "PhD"])
marital_status = st.selectbox("Marital Status", ["Single","Couple"])
income = st.number_input("Income (in $)", min_value=1000, max_value=1000000, value=50000)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=1000, value=30)
complain = st.selectbox("Has Complained?", ["No", "Yes"])
response = st.selectbox("Responded to Last Campaign?", ["No", "Yes"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tot_expenses = st.number_input("Total Expenses (in $)", min_value=0, max_value=100000, value=1000)
tot_accepted_cmp = st.number_input("Total Accepted Campaigns", min_value=0, max_value=10, value=0)
tot_purchases = st.number_input("Total Purchases", min_value=0, max_value=100, value=10)
tot_children = st.number_input("Total Children", min_value=0, max_value=10, value=1)
tot_adults = st.number_input("Total Adults", min_value=1, max_value=10, value=2)
customer_since = st.number_input("Customer Since (years)", min_value=0, max_value=50, value=5)

# Prepare input data
input_data = pd.DataFrame({
    "Education": [education],
    "Marital_Status": [marital_status],
    "Income": [income],
    "Recency": [recency],
    "Complain": [1 if complain == "Yes" else 0],
    "Response": [1 if response == "Yes" else 0],
    "Age": [age],
    "Tot_Expenses": [tot_expenses],
    "Tot_AcceptedCmp": [tot_accepted_cmp],
    "Tot_Purchases": [tot_purchases],
    "Tot_Children": [tot_children],
    "Tot_adults": [tot_adults],
    "Customer_Since": [customer_since]
})

# Preprocess input data
processed_input = preprocess_data(input_data)

# Predict personality and segment
if st.button("Analyze Customer"):
    try:
        prediction = model.predict(processed_input)
        personality = prediction[0]

        # Example segmentation logic (adjust as needed)
        if income > 50000 and tot_purchases > 50:
            segment = "High-Value Customer"
        elif age < 30 and response == 1:
            segment = "Young Engaged Customer"
        else:
            segment = "General Customer"

        # Display results
        st.success(f"Predicted Personality: {personality}")
        st.info(f"Customer Segment: {segment}")

        # Provide personalized marketing insights
        st.header("Marketing Recommendations")
        if segment == "High-Value Customer":
            st.write("Focus on loyalty programs and premium offerings.")
        elif segment == "Young Engaged Customer":
            st.write("Emphasize social media campaigns and trendy products.")
        else:
            st.write("Provide general promotions and discounts.")

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")

# Optional: Display dataset
st.header("Dataset Viewer")
if st.checkbox("Show Dataset"):
    try:
        dataset = pd.read_csv("marketing_campaign.csv")
        st.write(dataset.head())
    except FileNotFoundError:
        st.error("Dataset file not found.")
