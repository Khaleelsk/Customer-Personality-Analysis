import streamlit as st
import pandas as pd

# Title and Description
st.title("Customer Personality Analysis")
st.write("Analyze customer data to gain insights into purchasing behavior and campaign responses.")

# Load Data
@st.cache_data
def load_data(file_path):
    data = pd.read_excel(file_path, sheet_name="marketing_campaign")
    return data

data_file = "marketing_campaign.xlsx"
df = load_data(data_file)

# Display Dataset
st.subheader("Dataset")
st.write("Below is a preview of the dataset:")
st.dataframe(df.head())

# Summary Statistics
st.subheader("Summary Statistics")
if st.button("Show Summary"):
    st.write(df.describe())

# Missing Values
st.subheader("Missing Values")
missing = df.isnull().sum()
st.write("Number of missing values in each column:")
st.write(missing[missing > 0])

# Visualization: Income Distribution
st.subheader("Income Distribution")
st.write("Visualizing the income distribution of customers:")
income_col = df['Income'].dropna()
st.bar_chart(income_col)

# Response Analysis
st.subheader("Campaign Responses")
response_counts = df['Response'].value_counts()
st.write("Response distribution for the most recent campaign:")
st.bar_chart(response_counts)

# Filter Data
st.subheader("Filter Data")
selected_education = st.multiselect("Select Education Level:", options=df['Education'].unique())
selected_status = st.multiselect("Select Marital Status:", options=df['Marital_Status'].unique())

filtered_df = df.copy()
if selected_education:
    filtered_df = filtered_df[filtered_df['Education'].isin(selected_education)]
if selected_status:
    filtered_df = filtered_df[filtered_df['Marital_Status'].isin(selected_status)]

st.write("Filtered Dataset:")
st.dataframe(filtered_df)

# Save Processed Data
st.subheader("Download Processed Data")
if st.button("Download Data"):
    csv = filtered_df.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name="filtered_data.csv", mime="text/csv")

st.write("Thank you for using the Customer Personality Analysis app!")
