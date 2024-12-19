import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="BTCEMI_MIN Predictor",
    page_icon="📊",
    layout="wide"
)

# Add title and description
st.title("BTCEMI_MIN Prediction App")
st.markdown("""
This app predicts BTCEMI_MIN using Random Forest based on:
- BTCENEGUE
- Year
- Month
- Day
""")

@st.cache_data
def load_data():
    """Load and prepare the dataset"""
    df = pd.read_excel('data_reduced_dimensionality.xlsx')
    return df

# Load the data
try:
    df = load_data()
    st.success("Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Display sample of the dataset
st.subheader("Sample Data")
st.dataframe(df.head())

# Display basic statistics
st.subheader("Data Statistics")
st.dataframe(df.describe())

# Prepare features and target
X = df[['BTCENEGUE', 'Year', 'Month', 'Day']]
y = df['BTCEMI_MIN']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
@st.cache_resource
def train_model():
    """Train the Random Forest model"""
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    return rf_model

# Train the model
with st.spinner("Training the model..."):
    model = train_model()
    st.success("Model trained successfully!")

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display metrics
st.subheader("Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("Mean Squared Error", f"{mse:.2f}")
col2.metric("Root Mean Squared Error", f"{rmse:.2f}")
col3.metric("R² Score", f"{r2:.3f}")

# Feature importance plot
st.subheader("Feature Importance")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

fig = px.bar(
    feature_importance,
    x='feature',
    y='importance',
    title='Feature Importance in Random Forest Model'
)
st.plotly_chart(fig)

# Create prediction interface
st.subheader("Make Predictions")
col1, col2, col3, col4 = st.columns(4)

with col1:
    btcenegue = st.number_input("BTCENEGUE", 
                               min_value=float(df['BTCENEGUE'].min()),
                               max_value=float(df['BTCENEGUE'].max()),
                               value=float(df['BTCENEGUE'].mean()))

with col2:
    year = st.number_input("Year",
                          min_value=int(df['Year'].min()),
                          max_value=int(df['Year'].max()),
                          value=int(df['Year'].mean()))

with col3:
    month = st.number_input("Month",
                           min_value=1,
                           max_value=12,
                           value=6)

with col4:
    day = st.number_input("Day",
                         min_value=1,
                         max_value=31,
                         value=15)

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

# Add actual vs predicted plot
st.subheader("Actual vs Predicted Values")
actual_vs_predicted = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

fig = px.scatter(
    actual_vs_predicted,
    x='Actual',
    y='Predicted',
    title='Actual vs Predicted Values'
)
fig.add_trace(
    px.line(x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()]).data[0]
)
st.plotly_chart(fig)

# Add residual plot
st.subheader("Residual Plot")
residuals = y_test - y_pred
residual_df = pd.DataFrame({
    'Predicted': y_pred,
    'Residuals': residuals
})

fig = px.scatter(
    residual_df,
    x='Predicted',
    y='Residuals',
    title='Residual Plot'
)
fig.add_hline(y=0, line_dash="dash")
st.plotly_chart(fig)
