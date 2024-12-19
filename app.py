import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="BTCEMI_MIN Predictor",
    layout="wide"
)

# Add title and description
st.title("BTCEMI_MIN Prediction App")
st.write("This app predicts BTCEMI_MIN based on BTCENEGUE and date inputs.")

@st.cache_data
def load_data():
    """Load and preprocess the data"""
    df = pd.read_excel('data_reduced_dimensionality.xlsx')
    return df

# Load the data
try:
    df = load_data()
    
    # Prepare the data
    X = df[['BTCENEGUE', 'Year', 'Month', 'Day']]
    y = df['BTCEMI_MIN']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Create input form
    st.subheader("Enter Prediction Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
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
    
    with col2:
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
    
    # Add predict button
    if st.button("Predict BTCEMI_MIN"):
        # Prepare input data
        input_data = np.array([[btcenegue, year, month, day]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Display prediction
        st.subheader("Prediction Result")
        st.success(f"Predicted BTCEMI_MIN: {prediction:.2f}")
        
        # Show feature importance
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': ['BTCENEGUE', 'Year', 'Month', 'Day'],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.bar_chart(importance_df.set_index('Feature'))

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please make sure the data file is in the correct location and format.")

# Add footer with additional information
st.markdown("---")
st.markdown("""
* The model uses Random Forest Regression for predictions
* Input values are automatically scaled using StandardScaler
* Feature importance chart shows the relative importance of each input variable
""")
