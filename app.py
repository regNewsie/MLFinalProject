import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Set page configuration
st.set_page_config(
    page_title="BTCEMI_MIN Predictor",
    layout="wide"
)

# Title and description
st.title("🌲 Random Forest BTCEMI_MIN Predictor")
st.markdown("Predict BTCEMI_MIN values based on temporal features using Random Forest.")

# Function to load and prepare data
@st.cache_data
def load_data():
    data = pd.read_excel('data_reduced_dimensionality.xlsx')
    X = data[data.columns[:-1]]
    y = data[data.columns[-1]]
    return X, y, data

# Load data
X, y, data = load_data()

# Sidebar for model parameters
st.sidebar.header("Model Parameters")
n_estimators = st.sidebar.slider("Number of trees", 50, 300, 100, 10)
max_depth = st.sidebar.slider("Maximum depth", 3, 20, 10, 1)
min_samples_split = st.sidebar.slider("Minimum samples to split", 2, 10, 2, 1)

# Train model
@st.cache_resource
def train_model(X, y, n_estimators, max_depth, min_samples_split):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X, y)
    return model

model = train_model(X, y, n_estimators, max_depth, min_samples_split)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Model Analysis", "Data Visualization"])

with tab1:
    st.header("Make Predictions")
    
    # Input form for predictions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        year = st.number_input("Year", min_value=2010, max_value=2025, value=2010)
    with col2:
        month = st.number_input("Month", min_value=1, max_value=12, value=1)
    with col3:
        day = st.number_input("Day", min_value=1, max_value=31, value=1)
    
    # Make prediction
    if st.button("Predict"):
        input_data = pd.DataFrame([[year, month, day]], columns=['Year', 'Month', 'Day'])
        prediction = model.predict(input_data)[0]
        
        st.success(f"Predicted BTCEMI_MIN: {prediction:.2f}")
        
        # Show confidence interval (using prediction intervals from RF)
        predictions = []
        for estimator in model.estimators_:
            predictions.append(estimator.predict(input_data)[0])
        
        lower = np.percentile(predictions, 2.5)
        upper = np.percentile(predictions, 97.5)
        
        st.info(f"95% Prediction Interval: [{lower:.2f}, {upper:.2f}]")

with tab2:
    st.header("Model Analysis")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig_importance = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance'
    )
    st.plotly_chart(fig_importance)
    
    # Model performance metrics
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    mse = np.mean((y - y_pred) ** 2)
    mae = np.mean(np.abs(y - y_pred))
    
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("MSE", f"{mse:.2f}")
    col3.metric("MAE", f"{mae:.2f}")

with tab3:
    st.header("Data Visualization")
    
    # Time series plot
    fig_timeseries = px.scatter(
        data,
        x=data.index,
        y='BTCEMI_MIN',
        title='BTCEMI_MIN Over Time'
    )
    st.plotly_chart(fig_timeseries)
    
    # Actual vs Predicted
    fig_parity = px.scatter(
        x=y,
        y=y_pred,
        labels={'x': 'Actual', 'y': 'Predicted'},
        title='Actual vs Predicted Values'
    )
    fig_parity.add_trace(
        go.Scatter(
            x=[y.min(), y.max()],
            y=[y.min(), y.max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash')
        )
    )
    st.plotly_chart(fig_parity)

# Save model button
if st.sidebar.button("Save Model"):
    joblib.dump(model, 'random_forest_model.joblib')
    st.sidebar.success("Model saved as 'random_forest_model.joblib'")

# Add some documentation
st.sidebar.markdown("""
### How to use this app:
1. Adjust model parameters in the sidebar
2. Use the Prediction tab to make new predictions
3. Explore model performance in the Model Analysis tab
4. Visualize the data in the Data Visualization tab
""")
