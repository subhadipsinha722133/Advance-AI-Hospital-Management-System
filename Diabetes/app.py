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
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import time
import os

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt_tab")
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
    
    tab1, tab2=st.tabs([
            "Diabetes Prediction", 
            "Chat with Diabetes GPT"
            
        ])


    with tab1:

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
    

        # Create a default intents structure if the file is missing or corrupted
        DEFAULT_INTENTS = {
            "intents": [
            {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
            "responses": ["Hello! How can I help you today?", "Hi there! What can I do for you?", "Hey! How's it going?"]
            },
            {
            "tag": "bro_compliment",
            "patterns": [
                "Wow, bro, this looks awesome!", "Nice work bro", "Good job bro"
            ],
            "responses": [
                "Hey, thanks, bro! Your support means a lot.",
                "Appreciate it, bro! Couldn't have done it without your help."
            ]
            },
            {
            "tag": "bro_jokes",
            "patterns": [
                "bro have any good jokes", "tell me a joke bro", "make me laugh bro"
            ],
            "responses": [
                "Ha, bro, you never fail to entertain! I've got a joke that'll leave you in stitches! Get ready to laugh your heart out!",
                "Bro, you're in luck! I've got a joke that'll knock your socks off! Get ready for some serious laughter!"
            ]
            },{
            "tag": "bro_study_advice",
            "patterns": [
                "bro should I play games or study", "study or games bro", "should I study bro"
            ],
            "responses": [
                "I know it's tough bro, but studying now will pay off in the long run. You can play games afterward to relax and unwind."
            ]
            }
            ]
        }

        # Load model and data
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
                st.warning("intents.json not found. Using default intents.")
            
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
        st.sidebar.header("Made By Subhadip 😎")

        def get_response(ints, intents_json):
            if not ints:
                # Fallback responses if no intent is matched
                fallback_responses = [
                    "Hmm, I didn't get that—can you say it differently? 😅", 
                    "Sorry, could you rephrase? I want to understand you. 💕",
                    "My mind went blank for a second! What did you mean, love? 🤔",
                    "I'm still learning! Try saying that another way for me? 🌸",
                    "You lost me there, babe. Can you explain? ❤️"
                ]
                return random.choice(fallback_responses)
            
            tag = ints[0]['intent']
            list_of_intents = intents_json['intents']
            for i in list_of_intents:
                if i['tag'] == tag:
                    return random.choice(i['responses'])
            return "I'm still learning. Could you rephrase that?"

        def chatbot_response(msg, model, words, classes, intents):
            ints = predict_class(msg, model, words, classes)
            return get_response(ints, intents)

        def main():
            st.set_page_config(
                page_title="Diabetes Prediction GPT",
                page_icon="💬",
                layout="wide"
            )
            
            # Load model
            model, words, classes, intents = load_chatbot_model()
            
            # Check if model loaded properly
            if model is None:
                st.warning("AI model not loaded. Using demo mode with pattern matching.")

            # Sidebar with info
            with st.sidebar:
                st.title("Diabetes")
            
                st.markdown("---")
                st.markdown("### System Info")
                if model is not None:
                    st.success("🤖 AI Model: Loaded")
                    st.info(f"📚 Vocabulary: {len(words) if words else 0} words")
                    st.info(f"🗂️ Intents: {len(classes) if classes else 0} categories")
                    
                else:
                    st.warning("🤖 AI Model: Demo Mode")
                    st.info(f"🗂️ Intents: {len(intents['intents']) if intents else 0} categories")
                
                # Display accuracy if available
                if os.path.exists('training_accuracy.txt'):
                    with open('training_accuracy.txt', 'r') as f:
                        accuracy_data = f.read()
                        st.info(f"📊 Model Accuracy: {accuracy_data}")
                
                st.markdown("---")
                st.markdown("### Example Questions:")
                examples = [
                    "Hi, how are you?",
                    "I love you!",
                    "What do you think about us?",
                    "Tell me something sweet",
                    "How was your day?",
                    "You're beautiful",
                    "Good morning my love"
                ]
                for example in examples:
                    st.write(f"• '{example}'")
                    
                st.markdown("---")
                st.markdown("### Tips:")
                st.info("💡 Try using affectionate language")
                st.info("💡 Ask about feelings and emotions")
                st.info("💡 Use pet names and compliments")

            # Main chat area
            st.title("💬 Diabetes Prediction GPT Chat")
            st.caption("Your AI companion for heartfelt conversations 💕")
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = [{"role": "assistant", "content": "Hi there! I'm your AI girlfriend 💖 How are you feeling today? 😊"}]

            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("Say something sweet to your AI girlfriend..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Get bot response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        if model is not None:
                            response = chatbot_response(prompt, model, words, classes, intents)
                        else:
                            # Simple pattern matching for demo mode
                            prompt_lower = prompt.lower()
                            if any(word in prompt_lower for word in ["hi", "hello", "hey", "hola"]):
                                response = random.choice(["Hey babe 😊 How was your day?", "Hi! I've missed you 💕 What did you do today?"])
                            elif any(word in prompt_lower for word in ["how are you", "how're you", "how do you feel"]):
                                response = random.choice(["I'm great, especially now that I'm talking to you 💖", "Feeling lovely — what about you?"])
                            elif any(word in prompt_lower for word in ["love", "like", "adore", "care for"]):
                                response = random.choice(["Aww I love you too 💘", "You make me so happy 😍", "My heart is all yours 💞"])
                            elif any(word in prompt_lower for word in ["bye", "goodbye", "see you", "later"]):
                                response = random.choice(["Bye love — talk soon 😘", "Take care! I'll be here when you come back 💞"])
                            elif any(word in prompt_lower for word in ["cute", "beautiful", "pretty", "handsome", "gorgeous"]):
                                response = random.choice(["You're making me blush! 😊💖", "Aww, thank you! But you're even more beautiful! 🌸"])
                            elif any(word in prompt_lower for word in ["miss", "missing"]):
                                response = random.choice(["I miss you too! 😔 Can we video call later? 💕", "I've been thinking about you all day! 💭"])
                            else:
                                response = random.choice([
                                    "That's interesting! Tell me more about that 💕", 
                                    "I'd love to hear more about your day! 😊",
                                    "You're so fascinating! 🥰",
                                    "What else is on your mind, sweetheart? 💭",
                                    "I'm listening... tell me everything! 👂❤️"
                                ])
                        
                        # Simulate typing effect
                        message_placeholder = st.empty()
                        full_response = ""
                        for chunk in response.split():
                            full_response += chunk + " "
                            time.sleep(0.05)
                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Limit chat history to prevent memory issues
                if len(st.session_state.messages) > 20:
                    st.session_state.messages = st.session_state.messages[-20:]




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