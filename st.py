import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load data
data_url = "datasets/datas_pre_processed.csv"
data = pd.read_csv(data_url)
data.drop(columns=['Unnamed: 0'], inplace=True)

# Train-test split
features = data.drop(columns=['sta']) 
target = data['sta']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app layout
st.title("Supervised Learning Streamlit App")
st.write("## Trained Model for 'sta' Prediction")

# Calculate MSE
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error: {mse:.2f}")

# User interaction for input
st.write("Choose feature values for prediction:")
input_data = {}
for column in features.columns:
    min_value = int(data[column].min())
    max_value = int(data[column].max())
    input_data[column] = st.slider(column, min_value, max_value)

# Make prediction
input_df = pd.DataFrame(input_data, index=[0])
prediction = model.predict(input_df)
st.write(f"Prediction for 'sta': {prediction[0]:.2f}")

# Main function
def main():
    st.write("Streamlit App Logic")

# Run Streamlit app
if __name__ == "__main__":
    main()
