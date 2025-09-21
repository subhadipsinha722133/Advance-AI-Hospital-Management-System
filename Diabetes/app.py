import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 2rem 0;
    }
    .high-risk {
        color: #ff4b4b;
        font-weight: bold;
    }
    .low-risk {
        color: #00cc66;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data function
@st.cache_data
def load_and_preprocess_data():
    data = pd.read_csv('diabetes_prediction_dataset.csv')
    data.drop_duplicates(inplace=True)
    
    X = data.drop(columns='diabetes', axis=1)
    Y = data['diabetes']
    
    # Handle categorical variables
    le_gender = LabelEncoder()
    le_smoking = LabelEncoder()
    
    X["gender"] = le_gender.fit_transform(X["gender"])
    X["smoking_history"] = le_smoking.fit_transform(X["smoking_history"])
    
    # Oversample to handle class imbalance
    ros = RandomOverSampler(random_state=42)
    X, Y = ros.fit_resample(X, Y)
    
    return X, Y, le_gender, le_smoking

# Train model function
@st.cache_resource
def train_model(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=42
    )
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, Y_train)
    
    # Calculate accuracy
    X_test_prediction = rf_model.predict(X_test)
    test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
    
    return rf_model, test_data_accuracy

# Main app
def main():
    st.title("🏥 Diabetes Prediction App")
    st.markdown("""
    This app predicts the likelihood of diabetes based on health metrics.
    Please fill in the details below to get a prediction.
    """)
    
    # Load data and train model
    try:
        X, Y, le_gender, le_smoking = load_and_preprocess_data()
        model, accuracy = train_model(X, Y)
        
        st.sidebar.success(f"Model loaded successfully! (Accuracy: {accuracy:.2%})")
    except Exception as e:
        st.error(f"Error loading data or model: {str(e)}")
        return
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", options=["Female", "Male", "Other"])
            age = st.slider("Age", min_value=0, max_value=100, value=30)
            hypertension = st.radio("Hypertension", options=["No", "Yes"])
            heart_disease = st.radio("Heart Disease", options=["No", "Yes"])
        
        with col2:
            smoking_history = st.selectbox(
                "Smoking History", 
                options=["never", "current", "former", "ever", "not current", "No Info"]
            )
            bmi = st.slider("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
            hba1c_level = st.slider("HbA1c Level", min_value=3.0, max_value=15.0, value=5.7, step=0.1)
            blood_glucose_level = st.slider("Blood Glucose Level", min_value=50, max_value=300, value=100)
        
        submitted = st.form_submit_button("Predict Diabetes Risk")
    
    if submitted:
        # Prepare input data
        input_data = {
            'gender': le_gender.transform([gender])[0],
            'age': age,
            'hypertension': 1 if hypertension == "Yes" else 0,
            'heart_disease': 1 if heart_disease == "Yes" else 0,
            'smoking_history': le_smoking.transform([smoking_history])[0],
            'bmi': bmi,
            'HbA1c_level': hba1c_level,
            'blood_glucose_level': blood_glucose_level
        }
        
        # Convert to numpy array and reshape
        input_array = np.array(list(input_data.values())).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_array)[0]
        prediction_proba = model.predict_proba(input_array)[0]
        
        # Display results
        st.markdown("---")
        st.markdown("### Prediction Results")
        
        if prediction == 1:
            st.error("""
            🚨 **High Risk of Diabetes**
            
            The model predicts a high likelihood of diabetes. 
            Please consult with a healthcare professional for proper diagnosis and guidance.
            """)
        else:
            st.success("""
            ✅ **Low Risk of Diabetes**
            
            The model predicts a low likelihood of diabetes. 
            Maintain a healthy lifestyle with regular check-ups.
            """)
        
        # Show probability
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probability of Diabetes", f"{prediction_proba[1]:.2%}")
        with col2:
            st.metric("Probability of No Diabetes", f"{prediction_proba[0]:.2%}")
        
        # Health recommendations
        st.markdown("---")
        st.markdown("### Health Recommendations")
        
        if prediction == 1:
            st.warning("""
            **If you're at high risk:**
            - Schedule a doctor's appointment for proper testing
            - Monitor your blood sugar levels regularly
            - Maintain a healthy diet low in sugar and processed foods
            - Engage in regular physical activity
            - Maintain a healthy weight
            """)
        else:
            st.info("""
            **To maintain good health:**
            - Continue with regular health check-ups
            - Maintain a balanced diet
            - Exercise regularly (at least 30 minutes daily)
            - Avoid smoking and limit alcohol consumption
            - Manage stress levels
            """)

    # Add some information about the app
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About This App")
    st.sidebar.info("""
    This diabetes prediction app uses a Random Forest Classifier trained on 
    health data to assess the risk of diabetes. The model considers factors like:
    - Demographic information (age, gender)
    - Medical history (hypertension, heart disease)
    - Lifestyle factors (smoking, BMI)
    - Blood test results (HbA1c, glucose levels)
    
    **Note:** This is a predictive tool and not a substitute for professional medical advice.
    Always consult with healthcare professionals for proper diagnosis.
    """)

if __name__ == "__main__":
    main()