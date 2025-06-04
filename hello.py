import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the model
model = load_model("house_price_ann_model.keras", compile=False)
model.compile(optimizer='adam', loss='mean_squared_error')

# Load dataset (for fitting scaler)
df = pd.read_csv("synthetic_house_prices.csv")

# Features used during training
features = ['LotArea', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea',
            'FullBath', 'BedroomAbvGr', 'GarageCars', 'GarageArea']

# Fit the scaler on full dataset
scaler = MinMaxScaler()
scaler.fit(df[features])

# Streamlit UI
st.title("🏠 House Price Predictor (ANN Model)")
st.markdown("Enter the house features to estimate the price:")

user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"{feature}", min_value=0.0, value=0.0)

if st.button("Predict"):
    input_array = np.array([list(user_input.values())])
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)
    st.success(f"💰 Estimated Sale Price: ₹{prediction[0][0]:,.2f}")
