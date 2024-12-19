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

# Add warning about data limitations
st.warning("""
⚠️ Important Data Range Limitations:
- Year range: This model was trained on data from 2010 to 2011 only
- BTCENEGUE range: Values between 880,000 to 6.6 million only
Predictions outside these ranges may not be reliable!
""")

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

# Input fields with more context
btcenegue = st.number_input(
    "BTCENEGUE (Electricity Consumption)",
    min_value=880000.0,
    max_value=6600000.0,
    value=1000000.0,
    help="Enter value between 880,000 and 6.6 million"
)

year = st.number_input(
    "Year",
    min_value=2010,
    max_value=2011,
    value=2010,
    help="Model trained on years 2010-2011 only"
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

# Add dynamic warning for out-of-range values
if btcenegue < 880000 or btcenegue > 6600000:
    st.warning("⚠️ BTCENEGUE value is outside the trained data range!")
    
if year < 2010 or year > 2011:
    st.warning("⚠️ Year value is outside the trained data range!")

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
