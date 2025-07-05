House Price Predictor (ANN Model)
This is a simple machine learning web app built using Streamlit and TensorFlow/Keras. It predicts the estimated sale price of a house based on user-input features using an Artificial Neural Network (ANN) model trained on a synthetic dataset.

Features
Predicts house prices based on 8 key features:

LotArea

YearBuilt

TotalBsmtSF

GrLivArea

FullBath

BedroomAbvGr

GarageCars

GarageArea

Uses a trained ANN model saved as .keras

Scales input using MinMaxScaler for better prediction accuracy

Interactive web interface using Streamlit

📂 Project Structure
bash
Copy
Edit
house_price_ann_appp/
hello.py                      
house_price_ann_model.keras  
synthetic_house_prices.csv   

⚙️ How to Run Locally
Clone this repository:

bash
Copy
Edit
git clone https://github.com/HELLO123-ui/house-price-ann-predictor.git
cd house-price-ann-predictor
Create a virtual environment (optional but recommended):

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
(or install manually):

bash
Copy
Edit
pip install streamlit tensorflow scikit-learn pandas numpy
Run the app:

bash
Copy
Edit
streamlit run hello.py
📊 Model Details
Model Type: Sequential ANN

Framework: TensorFlow / Keras

Input Features: 8 normalized features

Output: Single scalar (predicted house price in INR)
