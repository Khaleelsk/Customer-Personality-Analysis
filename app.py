import streamlit as st
import pandas as pd
import pickle

# Preprocessing function
def preprocess_data(data):
    # Handle missing values if needed
    if 'Income' in data.columns:
        data['Income'].fillna(data['Income'].median(), inplace=True)
    
    # Encoding categorical features
    education_mapping = {'Undergradute': 0, 'Graduation': 1, 'Postgraduate': 2, 'PhD': 3}
    marital_status_mapping = {'Single': 0, 'Couple':1}

    data['Education'] = data['Education'].map(education_mapping)
    data['Marital_Status'] = data['Marital_Status'].map(marital_status_mapping)

    # Ensure the input data matches the training features
    required_features = [
        'Education', 'Marital_Status', 'Income', 'Recency', 'Complain',
        'Response', 'Age', 'Tot_Expenses', 'Tot_AcceptedCmp', 'Tot_Purchases',
        'Tot_Children', 'Tot_adults', 'Customer_Since'
    ]
    data = data[required_features]

    return data

# Load the trained model
@st.cache_resource
def load_model():
    with open("CPA_model.pkl", "rb") as file:
        return pickle.load(file)

# Streamlit app
st.title("Customer Personality Analysis")

st.write("""
Customer Personality Analysis is one the most important applications of unsupervised learning.Using clustering techniques, companies can identify the several segments of customers allowing them to target the potential user base. 
In this machine learning project, we will make use of K-means clustering which is the essential algorithm for clustering unlabeled dataset. Before ahead in this project, learn what actually Customer Personality Analysis is.

This app predicts customer personality traits based on input data.Provide the necessary inputs and get predictions!""")

# Load model
model = load_model()

# User input form
st.header("Enter Customer Details")

# User input fields
education = st.selectbox("Education", [ "Undergraduate","Graduation", "Postgraduate", "PhD"])
marital_status = st.selectbox("Marital Status", ["Single","Couple"])
income = st.number_input("Income ($)", min_value=1000, max_value=1000000, value=50000)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=100, value=30)
complain = st.selectbox("Has Complained?", ["No", "Yes"])
response = st.selectbox("Responded to Last Campaign?", ["No", "Yes"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tot_expenses = st.number_input("Total Expenses (in $)", min_value=0, max_value=3000, value=1000)
tot_accepted_cmp = st.number_input("Total Accepted Campaigns", min_value=0, max_value=5, value=0)
tot_purchases = st.number_input("Total Purchases", min_value=0, max_value=45, value=10)
tot_children = st.number_input("Total Children", min_value=0, max_value=10, value=1)
tot_adults = st.number_input("Total Adults", min_value=1, max_value=10, value=2)
family_size = st.number_input("Family Size", min_value=1, max_value=10, value=3)
customer_since = st.number_input("Customer Since (years)", min_value=0, max_value=700, value=5)

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
    "Family_size": [family_size],
    "Customer_Since": [customer_since]
})

# Preprocess input data
processed_input = preprocess_data(input_data)

# Predict
if st.button("Predict Personality"):
    try:
        prediction = model.predict(processed_input)
        if prediction==1:
            st.success(f"Predicted Personality: Customer is Active with the Campaigns.")
            if income<=30000:
                st.write("The Customer belongs the Lower Class.")
            elif income>30000 and income<=60000 :
                st.write("The Customer belongs the Middle Class.")
            else:
                st.write("The Customer belongs the High Class.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Optional: Display dataset
st.header("Dataset Viewer")
if st.checkbox("Show Dataset"):
    try:
        dataset = pd.read_csv("marketing_campaign.csv")
        st.write(dataset.head())
    except FileNotFoundError:
        st.error("Dataset file not found.")
