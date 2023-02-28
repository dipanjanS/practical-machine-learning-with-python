# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:56:29 2023

@author: HP
"""


import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('medical_insurance_cost_predictor.sav', 'rb'))

#creating a function for Prediction
def medical_insurance_cost_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    return prediction

def main():
    
    #giving a title
    st.title('Medical Insurance Prediction Web App')
    
    #getting input from the user
    
    age = st.text_input('Age')
    sex = st.text_input('Sex: 0 -> Female, 1 -> Male')
    bmi = st.text_input('Body Mass Index')
    children = st.text_input('Number of Children')
    smoker = st.text_input('Smoker: 0 -> No, 1 -> Yes')
    region = st.text_input('Region of Living: 0 -> NorthEast, 1-> NorthWest, 2-> SouthEast, 3-> SouthWest')
    
    #code for prediction
    diagnosis = ''
    
    # getting the input data from the user
    if st.button('Predicted Medical Insurance Cost: '):
        diagnosis = medical_insurance_cost_prediction([age,sex,bmi,children,smoker,region])
        
    st.success(diagnosis)
    

if __name__ == '__main__':
    main()
