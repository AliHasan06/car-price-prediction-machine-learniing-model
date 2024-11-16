import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Input dictionaries for categorical encoding
Model_dic = {'Ambassador': 0, 'Audi': 1, 'BMW': 2, 'Chevrolet': 3, 'Datsun': 4, 'Fiat': 5, 'Force': 6, 'Ford': 7, 'Honda': 8, 'Hyundai': 9, 'Isuzu': 10, 'Jaguar': 11, 'Jeep': 12, 'Kia': 13, 'Land': 14, 'Lexus': 15, 'Mahindra': 16, 'Maruti': 17, 'Mercedes-Benz': 18, 'Mitsubishi': 19, 'Nissan': 20, 'Renault': 21, 'Skoda': 22, 'Tata': 23, 'Toyota': 24, 'Volkswagen': 25}
trans_dict = {'Manual': 0, 'Automatic': 1}
engine_type_dic = {'LPG': 1, 'Diesel': 2, 'Petrol': 3}
owner_dict = {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 0}

# Model dict
ModelDictionary = {
    '3 Series 320d GT Luxury Line': 0,
    '3 Series 320d Luxury Line': 1,
    '5 Series 520d Sport Line': 2,
    '6 Series GT 630d Luxury Line': 3,
}

# List of categories
Model_list = list(Model_dic.keys())
engine_type_list = list(engine_type_dic.keys())
owner_list = list(owner_dict.keys())

st.set_page_config(page_title='Used Car Price Prediction by Ali Hasan', page_icon='🚗')

# Load dataset
car = pd.read_csv('Car_cleaned.csv')

# Function to filter Models by Model
def find_Model(Model):
    Models = car[car['Model'] == Model]['Model']
    return Models.tolist()

# Load prediction model
@st.cache_data  # Updated caching method
def model_loader(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"Model file not found at: {path}")
        return None

model_forest = model_loader("finalized_Model.pkl")

# Page layout
st.markdown("<h2 style='text-align: center;'>🚗  Used Car Price Prediction™  🚗</h2>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

# Input fields
mileage = col1.number_input('Enter mileage (e.g., 200 miles):', min_value=0, value=0, help='How much the car has been driven?')  # Added default value
year = col1.slider('Year of Manufacture:', 1980, 2020, 2005, help='When was the car manufactured?')
Model_inp = col1.selectbox('Select Model:', options=Model_list, help='Select the car Model.')

model_code = Model_dic[Model_inp]

engine_type_inp = col1.selectbox('Engine Type:', options=engine_type_list, help='Select fuel type.')
engine_type = engine_type_dic[engine_type_inp]

engineV = col2.number_input('Engine size (e.g., 660cc):', min_value=0.0, max_value=2500.0, value=0.0, step=0.1, help='Enter engine size in cc.')  # Added default value

# Filter models for the selected Model
Model_list = find_Model(Model_inp)
if Model_list:
    Model_inp = col2.selectbox('Select Model:', options=Model_list, help='Select car model.')
    Model = ModelDictionary.get(Model_inp, -1)
else:
    st.warning('No models available for this Model.')
    Model = -1

# Prediction
inp_array = np.array ([[mileage, engineV, year, model_code, engine_type, Model]])

predict = col1.button('Predict')

if predict:
    if Model == -1:
        st.error('Invalid model selection. Please choose a valid model.')
    else:
        pred = model_forest.predict(inp_array)
        if pred < 0:
            st.error('Invalid input. Please enter realistic values.')
        else:
            st.success(f'The predicted price of the car is ${round(float(pred), 3)} 🚙')
            st.balloons()

# Project info
st.header('🧭 About the Project')
st.write("""
This app predicts the price of used cars 🚙 based on various features like Model, mileage, year of manufacture, and engine type. 
Check out the full project on [GitHub](https://github.com/AliHasan06/car-price-prediction-machine-learning-Model). Contact: alihassanraja06@gmail.com 📫
""")