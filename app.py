import streamlit as st
import pickle
import pandas as pd
from streamlit_option_menu import option_menu

# Set page title and icon
st.set_page_config(page_title="Disease Prediction", page_icon="🩺")

# Hide Streamlit's default UI elements
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Adding background image
background_image_url = "background_image_url = https://images.unsplash.com/photo-1588776814546-1aadf618e0a7"

page_bg_img = f"""
<style> 
[data-testid="stAppViewContainer"] {{
    background-image: url({background_image_url});
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the trained models correctly
models = {
    'diabetes': pickle.load(open('Models/diabetes_model.sav', 'rb')),
    'heart_disease': pickle.load(open('Models/heart_disease_model.sav', 'rb')),
    'parkinsons': pickle.load(open('Models/parkinsons_model.sav', 'rb')),
    # 'lung_cancer': pickle.load(open('Models/dataset_lung_cancer.sav', 'rb'))
}

# Create a dropdown menu for disease prediction
selected = st.selectbox(
    'Select a Disease to Predict',
    ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction']
)

# Function to handle input fields
def display_input(label, tooltip, key, type="text"):
    if type == "text":
        return st.text_input(label, key=key, help=tooltip)
    elif type == "number":
        return st.number_input(label, key=key, help=tooltip, step=1)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction")
    st.write("Enter the following details to predict Parkinson's disease:")

    fo = display_input('MDVP:Fo(Hz)', 'Enter MDVP:Fo(Hz) value', 'fo', 'number')
    fhi = display_input('MDVP:Fhi(Hz)', 'Enter MDVP:Fhi(Hz) value', 'fhi', 'number')
    flo = display_input('MDVP:Flo(Hz)', 'Enter MDVP:Flo(Hz) value', 'flo', 'number')
    Jitter_percent = display_input('MDVP:Jitter(%)', 'Enter MDVP:Jitter(%) value', 'Jitter_percent', 'number')
    Jitter_Abs = display_input('MDVP:Jitter(Abs)', 'Enter MDVP:Jitter(Abs) value', 'Jitter_Abs', 'number')
    RAP = display_input('MDVP:RAP', 'Enter MDVP:RAP value', 'RAP', 'number')
    PPQ = display_input('MDVP:PPQ', 'Enter MDVP:PPQ value', 'PPQ', 'number')
    DDP = display_input('Jitter:DDP', 'Enter Jitter:DDP value', 'DDP', 'number')
    Shimmer = display_input('MDVP:Shimmer', 'Enter MDVP:Shimmer value', 'Shimmer', 'number')
    Shimmer_dB = display_input('MDVP:Shimmer(dB)', 'Enter MDVP:Shimmer(dB) value', 'Shimmer_dB', 'number')
    APQ3 = display_input('Shimmer:APQ3', 'Enter Shimmer:APQ3 value', 'APQ3', 'number')
    APQ5 = display_input('Shimmer:APQ5', 'Enter Shimmer:APQ5 value', 'APQ5', 'number')
    APQ = display_input('MDVP:APQ', 'Enter MDVP:APQ value', 'APQ', 'number')
    DDA = display_input('Shimmer:DDA', 'Enter Shimmer:DDA value', 'DDA', 'number')
    NHR = display_input('NHR', 'Enter NHR value', 'NHR', 'number')
    HNR = display_input('HNR', 'Enter HNR value', 'HNR', 'number')
    RPDE = display_input('RPDE', 'Enter RPDE value', 'RPDE', 'number')
    DFA = display_input('DFA', 'Enter DFA value', 'DFA', 'number')
    spread1 = display_input('Spread1', 'Enter Spread1 value', 'spread1', 'number')
    spread2 = display_input('Spread2', 'Enter Spread2 value', 'spread2', 'number')
    D2 = display_input('D2', 'Enter D2 value', 'D2', 'number')
    PPE = display_input('PPE', 'Enter PPE value', 'PPE', 'number')

    parkinsons_diagnosis = ''
    if st.button("Parkinson's Test Result"):
        if 'parkinsons' in models:  # Ensure the model exists
            import numpy as np
            features = np.array([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])
            
            # Debugging output
            st.write("Model expects", models['parkinsons'].n_features_in_, "features")
            st.write("Provided features:", features.shape[1])
            
            # Check if feature count matches
            expected_features = models['parkinsons'].n_features_in_
            if features.shape[1] != expected_features:
                st.error(f"Feature mismatch: Model expects {expected_features}, but got {features.shape[1]}")
            else:
                # Make the prediction
                parkinsons_prediction = models['parkinsons'].predict(features)
                parkinsons_diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"
                st.success(parkinsons_diagnosis)
        else:
            st.error("Parkinson's model not found! Make sure the model is loaded correctly.")


#Diabetes Prediction Page 
if selected == 'Diabetes Prediction': 
    st.title('Diabetes') 
    st.write(" Enter the following details to predict diabetes :")

    Pregnancies = display_input('Number of Pregnancies', 'Enter number of times pregnant', 'Pregnancies', 'number') 
    Glucose = display_input('Glucose Level', 'Enter glucose level', 'Glucose', 'number') 
    BloodPressure = display_input('Blood Pressure value', 'Enter blood pressure value', 'BloodPressure', 'number')
    SkinThickness = display_input('skin Thickness value', 'Enter skin thickness value', 'skinThickness', 'number')
    Insulin = display_input('Insulin Level','Enter insulin level', 'Insulin', 'number') 
    BMI = display_input('BMI value', 'Enter Body Mass Index value', 'BMI', 'number') 
    DiabetesPedigreeFunction = display_input('Diabetes Pedigree Function value', 'Enter diabetes pedigree function value', 'DiabetesPedigreeFunction','number')
    Age = display_input('Age of the Person', 'Enter age of the person', 'Age', 'number')

    diab_diagnosis = '' 
    if st.button('Diabetes Test Result'): 
        diab_prediction = models['diabetes'].predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic' 
        st.success(diab_diagnosis)

# #Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease') 
    st.write("Enter the following details to predict heart disease:") 
    
    age = display_input('Age', 'Enter age of the person', 'age', 'number') 
    sex = display_input('Sex (1 = male; 0 = female)', 'Enter sex of the person', 'sex', 'number') 
    cp = display_input('Chest Pain types (0, 1, 2, 3)', 'Enter chest pain type', 'cp', 'number') 
    trestbps = display_input('Resting Blood Pressure', 'Enter resting blood pressure', 'trestbps', 'number')
    chol = display_input('Serum Cholesterol in mg/dl', 'Enter serum cholesterol', 'chol', 'number') 
    fbs = display_input('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)', 'Enter fasting blood sugar', 'fbs', 'number') 
    restecg = display_input('Resting Electrocardiographic results (0, 1, 2)', 'Enter resting ECG results', 'restecg', 'number') 
    thalach = display_input('Maximum Heart Rate achieved', 'Enter maximum heart rate', 'thalach', 'number')
    exang = display_input('Exercise Induced Angina (1 = yes; 0 = no)', 'Enter exercise induced angina', 'exang', 'number') 
    oldpeak = display_input('ST depression induced by exercise', 'Enter ST depression value', 'oldpeak', 'number') 
    slope = display_input('Slope of the peak exercise ST segment (0, 1, 2)', 'Enter slope value', 'slope', 'number') 
    ca = display_input('Major vessels colored by fluoroscopy (0-3)', 'Enter number of major vessels', 'ca', 'number') 
    thal = display_input('Thal (0 = normal; 1 = fixed defect; 2 = reversible defect)', 'Enter thal value', 'thal', 'number')

    heart_diagnosis = '' 
    if st.button('Heart Disease Test Result'):
        if 'heart_disease' in models:
            import numpy as np
            features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            
            # Debugging output
            st.write("Model expects", models['heart_disease'].n_features_in_, "features")
            st.write("Provided features:", features.shape[1])
            
            # Check if feature count matches
            expected_features = models['heart_disease'].n_features_in_
            if features.shape[1] != expected_features:
                st.error(f"Feature mismatch: Model expects {expected_features}, but got {features.shape[1]}")
            else:
                heart_prediction = models['heart_disease'].predict(features)
                heart_diagnosis = "The person has heart disease" if heart_prediction[0] == 1 else "The person does not have heart disease"
                st.success(heart_diagnosis)
        else:
            st.error("Heart disease model not found! Make sure the model is loaded correctly.")
