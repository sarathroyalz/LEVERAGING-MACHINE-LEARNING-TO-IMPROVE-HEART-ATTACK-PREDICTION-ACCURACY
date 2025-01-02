import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load your dataset
# Replace 'your_dataset.csv' with the path to your actual dataset
df = pd.read_csv('your_dataset.csv')

# Specify your feature columns and target variable
# Adjust 'feature1', 'feature2', ..., 'target' to your actual column names
features = df[['chest pain type', 'resting blood pressure', 'serum cholestoral', 
                'fasting blood sugar', 'resting electrocardiographic results', 
                'maximum heart rate', 'exercise induced angina']]
target = df['heart disease']  # Replace with your target column name

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)

# Train the model
model.fit(X_train_scaled, y_train)

# Save the scaler to a file
with open('sc.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Save the model to a file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model and scaler saved successfully.")
