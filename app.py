import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
data = pd.read_excel('data_reduced_dimensionality.xlsx')

# Separate features and target
X = data[data.columns[:-1]]
y = data[data.columns[-1]]

# Initialize models
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'Linear Regression': LinearRegression(),
    'SVR': SVR()
}

# Initialize lists to store R² scores
train_scores = []
test_scores = []
model_names = []

# Perform k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    train_r2_scores = []
    test_r2_scores = []
    
    for train_idx, test_idx in kf.split(X):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Get predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate R² scores
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        train_r2_scores.append(train_r2)
        test_r2_scores.append(test_r2)
    
    # Store scores
    train_scores.extend(train_r2_scores)
    test_scores.extend(test_r2_scores)
    model_names.extend([name] * len(train_r2_scores))

# Create DataFrame for plotting
plot_data = pd.DataFrame({
    'Model': model_names * 2,
    'R² Score': train_scores + test_scores,
    'Type': ['Training'] * len(train_scores) + ['Testing'] * len(test_scores)
})

# Create boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=plot_data, x='Model', y='R² Score', hue='Type')
plt.xticks(rotation=45)
plt.title('Training vs Testing R² Scores Across Models')
plt.tight_layout()
plt.show()

# Print mean scores
print("\nMean R² Scores:")
for name in models.keys():
    train_mask = (plot_data['Model'] == name) & (plot_data['Type'] == 'Training')
    test_mask = (plot_data['Model'] == name) & (plot_data['Type'] == 'Testing')
    
    mean_train = plot_data[train_mask]['R² Score'].mean()
    mean_test = plot_data[test_mask]['R² Score'].mean()
    
    print(f"\n{name}:")
    print(f"Training R² (mean): {mean_train:.3f}")
    print(f"Testing R² (mean): {mean_test:.3f}")
