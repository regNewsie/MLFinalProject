import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Set page configuration
st.set_page_config(
    page_title="BTCEMI_MIN Predictor",
    layout="centered"
)

# Add title
st.title("BTCEMI_MIN Prediction")

# Load and prepare data
@st.cache_data
def load_and_train_model():
    # Load data
    df = pd.read_excel('data_reduced_dimensionality.xlsx')
    
    # Prepare features and target
    X = df[['BTCENEGUE', 'Year', 'Month', 'Day']]
    y = df['BTCEMI_MIN']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, df

try:
    model, df = load_and_train_model()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Create prediction interface
st.subheader("Enter Values for Prediction")

# Input fields
btcenegue = st.number_input(
    "BTCENEGUE",
    min_value=float(df['BTCENEGUE'].min()),
    max_value=float(df['BTCENEGUE'].max()),
    value=float(df['BTCENEGUE'].mean())
)

year = st.number_input(
    "Year",
    min_value=int(df['Year'].min()),
    max_value=int(df['Year'].max()),
    value=int(df['Year'].mean())
)

month = st.number_input(
    "Month",
    min_value=1,
    max_value=12,
    value=6
)

day = st.number_input(
    "Day",
    min_value=1,
    max_value=31,
    value=15
)

# Make prediction
if st.button("Predict BTCEMI_MIN"):
    input_data = pd.DataFrame({
        'BTCENEGUE': [btcenegue],
        'Year': [year],
        'Month': [month],
        'Day': [day]
    })
    
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted BTCEMI_MIN: {prediction:.2f}")
