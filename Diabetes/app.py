import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import json
import random
import time
import os

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

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
    .chat-container {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        height: 500px;
        overflow-y: auto;
        margin-bottom: 20px;
        border: 1px solid #ddd;
    }
    .user-message {
        background-color: #dcf8c6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: right;
        margin-left: 20%;
    }
    .bot-message {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: left;
        margin-right: 20%;
        border: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data function for diabetes prediction
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

# Train model function for diabetes prediction
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

# Create a default intents structure if the file is missing or corrupted
DEFAULT_INTENTS = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
            "responses": ["Hello! How can I help you with diabetes information today?", 
                         "Hi there! I'm here to answer your diabetes-related questions.", 
                         "Hey! How can I assist you with diabetes information?"]
        },
        {
            "tag": "diabetes_question",
            "patterns": ["What is diabetes", "Tell me about diabetes", "Explain diabetes"],
            "responses": ["Diabetes is a chronic condition that affects how your body turns food into energy. There are several types, with type 2 being the most common."]
        },
        {
            "tag": "symptoms",
            "patterns": ["What are symptoms of diabetes", "Signs of diabetes", "How do I know if I have diabetes"],
            "responses": ["Common symptoms include frequent urination, excessive thirst, extreme hunger, unexplained weight loss, and fatigue."]
        },
        {
            "tag": "prevention",
            "patterns": ["How to prevent diabetes", "Can I prevent diabetes", "Avoid diabetes"],
            "responses": ["You can reduce your risk by maintaining a healthy weight, eating a balanced diet, exercising regularly, and avoiding smoking."]
        },
        {
            "tag": "thanks",
            "patterns": ["Thank you", "Thanks", "That's helpful", "Appreciate it"],
            "responses": ["You're welcome! Feel free to ask more questions about diabetes.", 
                         "Happy to help! Let me know if you have other questions."]
        }
    ]
}

# Load chatbot model and data
@st.cache_resource
def load_chatbot_model():
    # Check if model files exist in the current directory
    model_path = 'Diabetes_model.h5'
    words_path = 'Diabetes_words.pkl'
    classes_path = 'Diabetes_classes.pkl'
    intents_path = 'intents.json'
    
    # Check if files exist
    missing_files = []
    if not os.path.exists(model_path):
        missing_files.append('Diabetes_model.h5')
    if not os.path.exists(words_path):
        missing_files.append('Diabetes_words.pkl')
    if not os.path.exists(classes_path):
        missing_files.append('Diabetes_classes.pkl')
    if not os.path.exists(intents_path):
        missing_files.append('intents.json')
        st.warning("intents.json not found. Using default diabetes-related intents.")
    
    # Try to load model files with error handling
    model = None
    words = None
    classes = None
    intents = DEFAULT_INTENTS
    
    try:
        if os.path.exists(model_path):
            model = load_model(model_path)
        if os.path.exists(words_path):
            with open(words_path, 'rb') as f:
                words = pickle.load(f)
        if os.path.exists(classes_path):
            with open(classes_path, 'rb') as f:
                classes = pickle.load(f)
        
        # Load intents from JSON file with encoding handling
        if os.path.exists(intents_path):
            try:
                with open(intents_path, 'r', encoding='utf-8') as file:
                    intents = json.load(file)
            except UnicodeDecodeError:
                try:
                    with open(intents_path, 'r', encoding='latin-1') as file:
                        intents = json.load(file)
                except:
                    with open(intents_path, 'r', encoding='cp1252') as file:
                        intents = json.load(file)
            except json.JSONDecodeError:
                st.error("intents.json is corrupted. Using default intents.")
        
        return model, words, classes, intents
        
    except Exception as e:
        st.error(f"Error loading model or data: {str(e)}")
        return None, None, None, DEFAULT_INTENTS

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    if words is None:
        return np.array([])
        
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model, words, classes):
    if model is None or words is None or classes is None:
        return []
        
    p = bow(sentence, words, show_details=False)
    if len(p) == 0:
        return []
        
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": float(r[1])})
    return return_list

def get_response(ints, intents_json):
    if not ints:
        # Fallback responses if no intent is matched
        fallback_responses = [
            "I'm not sure I understand. Could you rephrase your question about diabetes?",
            "I'm specialized in diabetes information. Could you ask me something related to that?",
            "I didn't quite get that. Could you ask about diabetes symptoms, prevention, or management?",
            "I'm here to help with diabetes-related questions. What would you like to know?",
            "Could you try asking that differently? I'm best at answering diabetes questions."
        ]
        return random.choice(fallback_responses)
    
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm still learning about diabetes. Could you ask something else?"

def chatbot_response(msg, model, words, classes, intents):
    ints = predict_class(msg, model, words, classes)
    return get_response(ints, intents)

# Main app
def main():
    st.title("🏥 Diabetes Prediction & Information App")
    
    tab1, tab2 = st.tabs(["Diabetes Prediction", "Diabetes Information Chat"])
    
    with tab1:
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


    with tab2:
        st.markdown("### 💬 Diabetes Information Chat")
        st.caption("Ask me questions about diabetes symptoms, prevention, and management")
        
        # Load chatbot model
        chatbot_model, chatbot_words, chatbot_classes, chatbot_intents = load_chatbot_model()
        
        # Initialize chat history
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [
                {"role": "assistant", "content": "Hi! I'm here to answer your questions about diabetes. What would you like to know?"}
            ]
        
        # Display chat messages
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.chat_messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask a question about diabetes..."):
            # Add user message to chat history
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            # Get bot response
            with st.spinner("Thinking..."):
                if chatbot_model is not None:
                    response = chatbot_response(prompt, chatbot_model, chatbot_words, chatbot_classes, chatbot_intents)
                else:
                    # Simple pattern matching for demo mode
                    prompt_lower = prompt.lower()
                    if any(word in prompt_lower for word in ["hi", "hello", "hey", "hola"]):
                        response = "Hello! I can answer questions about diabetes. What would you like to know?"
                    elif any(word in prompt_lower for word in ["what is diabetes", "tell me about diabetes"]):
                        response = "Diabetes is a chronic condition that affects how your body processes blood sugar (glucose)."
                    elif any(word in prompt_lower for word in ["symptoms", "signs"]):
                        response = "Common symptoms include frequent urination, excessive thirst, unexplained weight loss, and fatigue."
                    elif any(word in prompt_lower for word in ["prevent", "avoid", "reduce risk"]):
                        response = "You can reduce diabetes risk by maintaining a healthy weight, eating balanced meals, and exercising regularly."
                    elif any(word in prompt_lower for word in ["thank", "thanks", "appreciate"]):
                        response = "You're welcome! Feel free to ask more questions about diabetes."
                    else:
                        response = "I'm specialized in diabetes information. Could you ask me something related to that?"
            
            # Add assistant response to chat history
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            
            # Rerun to update the chat display
            st.rerun()
        
        # Sidebar info for chatbot
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Chat Information")
        if chatbot_model is not None:
            st.sidebar.success("🤖 AI Model: Loaded")
        else:
            st.sidebar.warning("🤖 AI Model: Using simple pattern matching")
        
        st.sidebar.markdown("**Try asking:**")
        st.sidebar.markdown("- What is diabetes?")
        st.sidebar.markdown("- What are the symptoms?")
        st.sidebar.markdown("- How can I prevent diabetes?")
        st.sidebar.markdown("- What foods should I avoid?")

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