# 🧠 Machine Learning-Based Stroke Prediction Model for Early Detection and Prevention
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![Accuracy](https://img.shields.io/badge/Accuracy-81%25-brightgreen)
![University](https://img.shields.io/badge/University-SLIIT-purple)

## 📊 Research Overview
This research project implements a **stroke risk prediction system** using an **Artificial Neural Network (ANN)** model integrated with a user-friendly **Streamlit web interface**. The system analyzes personal health metrics to assess an individual's risk of experiencing a stroke, providing a valuable tool for preventive healthcare.

> **B.Sc. Research Project in Electrical and Electronic Engineering - Sri Lanka Institute of Information Technology (SLIIT)**

---

## 🔍 Research Highlights
### ✅ Dataset Characteristics
- **8,600 health records** collected from hospital data
- **2,500 stroke cases** identified for analysis
- Comprehensive patient health profiles including medical history and lifestyle factors

### ✅ Advanced Neural Network Architecture
- Implemented a **deep neural network** with multiple hidden layers (1024→512→256→128→64→32→16→1)
- Incorporated **dropout layers (0.2)** to prevent overfitting
- Applied **L2 regularization** for model stability
- Utilized **Adam optimizer** with fine-tuned learning rate

### ✅ Data Processing & Model Training
- **Preprocessed healthcare dataset** (categorical encoding, outlier handling)
- Resolved **data imbalance** using SMOTE (Synthetic Minority Over-sampling Technique)
- Implemented **feature scaling** using StandardScaler
- Achieved **81% accuracy** in stroke prediction

### ✅ Model Performance Metrics
- **Precision: ~81%** - High reliability in positive predictions
- **Recall: ~82%** - Strong capability to identify actual stroke cases
- **F1-Score: ~81-82%** - Balanced performance between precision and recall

### ✅ Interactive Web Application
- Developed a **responsive Streamlit interface** for user interaction
- Created an **intuitive health information form** with appropriate input constraints
- Implemented **real-time risk assessment** with visual feedback
- Displayed prediction results with **probability scores**

---

## 💻 Technologies Used
- **Python** - Core programming language
- **TensorFlow/Keras** - Neural network implementation
- **Pandas** - Data manipulation and preprocessing
- **NumPy** - Numerical operations
- **Scikit-learn** - Machine learning utilities (SMOTE, StandardScaler)
- **Matplotlib/Seaborn** - Data visualization
- **Streamlit** - Web application framework
- **Pickle** - Model serialization

---

## 📋 Key Health Indicators Analyzed
The ANN model processes the following user health metrics to predict stroke risk:

| Feature | Type | Description | Significance |
|---------|------|-------------|-------------|
| Gender | Categorical | Male/Female | Different risk profiles by gender |
| Age | Numerical | Age in years | Increasing risk with age |
| Hypertension | Binary | 0=No, 1=Yes | Major stroke risk factor |
| Heart Disease | Binary | 0=No, 1=Yes | Comorbidity affecting stroke risk |
| Avg. Glucose Level | Numerical | Blood glucose level (mg/dL) | Diabetes-related risk indicator |
| BMI | Numerical | Body Mass Index | Weight-related risk factor |
| Smoking Status | Categorical | Never smoked/Formerly smoked/Smokes | Lifestyle risk factor |

---

## 🖼️ Application Interface
The interactive Streamlit application provides an intuitive interface for users to input their health data and receive stroke risk predictions.

![Stroke Prediction App Interface](/app_interface.png)

---

## 🚀 Installation & Usage
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/stroke-prediction-system.git
   cd stroke-prediction-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

4. **Access the web interface**
   Open your browser and navigate to `http://localhost:8501`

---

## 📈 Model Performance
| Metric | Value |
|--------|-------|
| Accuracy | 81% |
| Precision | 81% |
| Recall | 82% |
| F1-Score | 81-82% |

---

## 📚 Implementation Details
The system workflow consists of three main components:

1. **Data Preprocessing Pipeline**
   - Categorical feature encoding
   - Missing value imputation
   - Feature scaling
   - Class imbalance handling via SMOTE

2. **Neural Network Model**
   ```python
   model = Sequential([
       Dense(1024, activation='relu', kernel_regularizer=l2(0.01)),
       Dropout(0.2),
       Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
       Dropout(0.2),
       # Additional layers...
       Dense(1, activation='sigmoid')
   ])
   ```

3. **Web Application Integration**
   - User data collection
   - Real-time prediction
   - Result visualization

---

## 🔮 Future Research Directions
- [ ] **Model Refinement**: Further optimization of neural network architecture
- [ ] **Hyperparameter Tuning**: Systematic exploration of optimal parameters
- [ ] **Clinical Deployment**: Testing in real-world healthcare settings
- [ ] **Feature Expansion**: Incorporate additional health parameters for improved prediction
- [ ] **Explainability**: Implement feature importance visualization for clinical interpretation
- [ ] **Personalized Recommendations**: Add tailored health advice based on risk factors
- [ ] **Mobile Application**: Develop companion app for wider accessibility
- [ ] **Longitudinal Analysis**: Implement user accounts for tracking health metrics over time

---

## 🌟 Research Impact
This research contributes to AI-driven healthcare solutions by:

- **Enhancing Early Detection**: Identifying high-risk individuals before symptom onset
- **Supporting Preventive Care**: Enabling targeted interventions for at-risk populations
- **Reducing Healthcare Burden**: Potential reduction in emergency care costs through prevention
- **Democratizing Healthcare**: Providing accessible stroke risk assessment tools
