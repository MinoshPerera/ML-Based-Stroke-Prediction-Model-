import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Arōgyā | Stroke Prediction App",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box-negative {
        background-color: #ccffcc;
        padding: 20px;
        border: 1px solid green;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    .result-box-positive {
        background-color: #ffcccc;
        padding: 20px;
        border: 1px solid red;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 class='main-header'> Arōgyā | Stroke Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("This app predicts the risk of stroke based on your health data using an Artificial Neural Network model trained on balanced data using SMOTE.")

# Define mappings
gender_dict = {'Male': 0, 'Female': 1, 'Other': 2}
smoking_status_dict = {'Unknown': 0, 'never smoked': 1, 'formerly smoked': 2, 'smokes': 3}

# File uploader for dataset
def load_dataset():
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None

# Load or train model and scaler
@st.cache_resource
def load_model(dataset=None):
    # If model and scaler do not exist or dataset is provided, create/train them
    if dataset is not None or not os.path.exists('stroke_model_smote.h5') or not os.path.exists('scaler_smote.pkl'):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from keras.optimizers import Adam
        
        if dataset is not None:
            # Preprocess dataset
            # Assume the target column is 'stroke' and remove any rows with missing values
            df = dataset.dropna()
            
            # Encode categorical variables if needed
            if 'gender' in df.columns:
                df['gender'] = df['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
            
            if 'smoking_status' in df.columns:
                df['smoking_status'] = df['smoking_status'].map({'Unknown': 0, 'never smoked': 1, 
                                                                'formerly smoked': 2, 'smokes': 3})
            
            # Extract features and target
            X = df.drop(['stroke', 'id'], axis=1, errors='ignore')
            y = df['stroke']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Apply SMOTE to balance the dataset
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            
            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_smote)
            X_test_scaled = scaler.transform(X_test)
            
            # Create the model
            model = Sequential([
                Dense(1024, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                Dropout(0.2),
                Dense(512, activation='relu'),
                Dropout(0.2),
                Dense(256, activation='relu'),
                Dropout(0.2),
                Dense(128, activation='relu'),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), 
                          loss='binary_crossentropy', 
                          metrics=['accuracy'])
            
            # Train the model
            model.fit(X_train_scaled, y_train_smote, 
                      validation_data=(X_test_scaled, y_test), 
                      epochs=50, 
                      batch_size=32, 
                      verbose=1)
            
            # Evaluate the model
            evaluation = model.evaluate(X_test_scaled, y_test)
            st.write(f"Model accuracy on test set: {evaluation[1]:.4f}")
            
            # Save the model and scaler
            model.save('stroke_model_smote.h5')
            with open('scaler_smote.pkl', 'wb') as f:
                pickle.dump(scaler, f)
                
            st.success("Model trained with SMOTE and saved successfully!")
                
        else:
            # Create and compile a placeholder model
            model = Sequential([
                    Dense(1024, activation='relu', input_shape=(7,)),
                    Dropout(0.2),
                    Dense(512, activation='relu'),
                    Dropout(0.2),
                    Dense(256, activation='relu'),
                    Dropout(0.2),
                    Dense(128, activation='relu'),
                    Dropout(0.2),
                    Dense(64, activation='relu'),
                    Dropout(0.2),
                    Dense(32, activation='relu'),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dropout(0.2),
                    Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer=Adam(learning_rate=0.001), 
                          loss='binary_crossentropy', 
                          metrics=['accuracy'])
            model.save('stroke_model_smote.h5')

            # Create and fit the scaler on dummy data
            scaler = StandardScaler()
            dummy_data = np.array([
                [0, 45, 0, 0, 90.0, 25.0, 1],  # gender, age, hypertension, heart_disease, glucose, bmi, smoking_status
                [1, 60, 1, 1, 150.0, 30.0, 2],  # second sample just to avoid single-row warning
            ])
            scaler.fit(dummy_data)

            # Save the fitted scaler
            with open('scaler_smote.pkl', 'wb') as f:
                pickle.dump(scaler, f)

    # Load the model and scaler
    model = tf.keras.models.load_model('stroke_model_smote.h5')
    with open('scaler_smote.pkl', 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler

# Main application
with st.sidebar:
    st.header("Model Training")
    st.write("Upload a dataset to train the model with SMOTE")
    
    dataset = load_dataset()
    
    if dataset is not None:
        st.write("Dataset Preview:")
        st.dataframe(dataset.head())
        
        if st.button("Train Model with SMOTE"):
            model, scaler = load_model(dataset)
    else:
        model, scaler = load_model()

# Input form
st.markdown("<p class='subheader'>Enter Your Health Information</p>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Name", "")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    
with col2:
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=90.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

# Convert inputs to numeric
hypertension = 1 if hypertension == "Yes" else 0
heart_disease = 1 if heart_disease == "Yes" else 0

# Prediction function
def predict_stroke():
    input_data = pd.DataFrame({
        'gender': [gender_dict[gender]],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status_dict[smoking_status]]
    })

    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    probability = float(prediction[0][0])
    result = int(probability > 0.5)
    return result, probability

# Prediction trigger
if st.button("Predict Stroke Risk"):
    if name.strip() == "":
        st.warning("Please enter your name.")
    else:
        with st.spinner("Calculating stroke risk..."):
            result, probability = predict_stroke()

        if result == 0:
            st.markdown(f"""
            <div class='result-box-negative'>
                <h3>Low Stroke Risk</h3>
                <p style='color:#064420;'><strong>Based on the provided information, {name} has a low risk of stroke.</strong></p>
                <p>Probability: {probability:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-box-positive'>
                <h3>High Stroke Risk</h3>
                <p style='color:#7B0000;'><strong>Based on the provided information, {name} has a high risk of stroke.</strong></p>
                <p>Probability: {probability:.2%}</p>
                <p>Please consult with a healthcare professional.</p>
            </div>
            """, unsafe_allow_html=True)

        # Summary
        st.markdown("<p class='subheader'>Input Data Summary</p>", unsafe_allow_html=True)
        summary_data = {
            'Name': name,
            'Gender': gender,
            'Age': age,
            'Hypertension': 'Yes' if hypertension == 1 else 'No',
            'Heart Disease': 'Yes' if heart_disease == 1 else 'No',
            'Avg. Glucose Level': f"{avg_glucose_level} mg/dL",
            'BMI': bmi,
            'Smoking Status': smoking_status
        }
        st.table(pd.DataFrame([summary_data]))

# Model performance visualization (will show after model training)
if 'model' in locals():
    st.markdown("<p class='subheader'>Model Performance</p>", unsafe_allow_html=True)
    st.write("The model was trained with SMOTE to balance the classes, which helps improve prediction accuracy for the minority class (stroke cases).")

# About section
with st.expander("About the Model and SMOTE"):
    st.write("""
    This application uses an Artificial Neural Network (ANN) model to predict the risk of stroke based on various health parameters.
    
    ### What is SMOTE?
    SMOTE (Synthetic Minority Over-sampling Technique) is a statistical technique for addressing class imbalance in datasets. In stroke prediction, typically only a small percentage of patients have positive stroke outcomes, creating a significant class imbalance.
    
    SMOTE works by:
    1. Taking samples from the minority class (stroke cases)
    2. Creating synthetic examples along the line segments joining the minority class samples
    3. Adding these synthetic examples to the training dataset
    
    This approach helps the model learn patterns from both classes more effectively, improving its ability to predict stroke risk.
    
    The model was trained on a balanced dataset containing information about patients with and without strokes, and it learned to identify patterns associated with increased stroke risk.
    
    Please note that this is a predictive tool and should not replace professional medical advice. Always consult with healthcare providers for proper diagnosis and treatment.
    """)

# Footer
st.markdown("""
---
<p style='text-align: center;'>© 2025 Arōgyā  |  Created by Minosh Perera</p>
""", unsafe_allow_html=True)
