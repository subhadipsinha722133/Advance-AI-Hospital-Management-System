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
import re
from difflib import get_close_matches

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
        background-color: #f0f8;
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
        background-color: #f99;
        border-radius: 10px;
        padding: 15px;
        height: 500px;
        overflow-y: auto;
        margin-bottom: 20px;
        border: 1px solid #ddd;
    }
    .user-message {
        background-color: #42a5f5;
        padding: 12px 15px;
        border-radius: 15px 15px 5px 15px;
        margin: 8px 0;
        color: black;
        margin-left: 20%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        word-wrap: break-word;
    }
    .bot-message {
        background-color: #37474f;
        padding: 12px 15px;
        border-radius: 15px 15px 15px 5px;
        margin: 8px 0;
        margin-right: 20%;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        word-wrap: break-word;
        
    }
    .quick-reply-btn {
        display: inline-block;
        background-color: #2E86AB;
        color: white;
        padding: 6px 12px;
        border-radius: 15px;
        margin: 3px;
        cursor: pointer;
        font-size: 12px;
        border: none;
        transition: background-color 0.3s;
    }
    .quick-reply-btn:hover {
        background-color: #1a5f7a;
    }
    .typing-indicator {
        font-style: italic;
        color: #666;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced diabetes knowledge base with comprehensive patterns and responses
ENHANCED_INTENTS = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": [
                "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
                "how are you", "what's up", "greetings", "hola", "howdy", "sup"
            ],
            "responses": [
                "Hello! I'm your diabetes information assistant. How can I help you today? 🏥",
                "Hi there! I'm here to answer your diabetes-related questions. What would you like to know?",
                "Hey! I'm specialized in diabetes information. Ask me anything about diabetes prevention, symptoms, or management! 💪",
                "Good day! I'm here to help with all your diabetes questions. What's on your mind?"
            ],
            "quick_replies": ["What is diabetes?", "Symptoms", "Prevention tips", "Diet advice"]
        },
        {
            "tag": "what_is_diabetes",
            "patterns": [
                "what is diabetes", "define diabetes", "explain diabetes", "tell me about diabetes",
                "diabetes definition", "what does diabetes mean", "diabetes basics", "diabetes 101"
            ],
            "responses": [
                """Diabetes is a chronic health condition that affects how your body turns food into energy. Here are the key types:

🔸 **Type 1 Diabetes**: Usually diagnosed in children and young adults. The body doesn't produce insulin.
🔸 **Type 2 Diabetes**: Most common form (90-95%). The body doesn't use insulin well.
🔸 **Gestational Diabetes**: Develops during pregnancy.

When you eat, your body breaks food down into sugar (glucose) and releases it into your bloodstream. Insulin helps glucose enter your cells to be used for energy. With diabetes, your body either doesn't make enough insulin or can't use it effectively."""
            ],
            "quick_replies": ["Types of diabetes", "Symptoms", "Risk factors", "Prevention"]
        },
        {
            "tag": "types_diabetes",
            "patterns": [
                "types of diabetes", "diabetes types", "different kinds of diabetes",
                "how many types diabetes", "type 1 vs type 2", "gestational diabetes"
            ],
            "responses": [
                """There are three main types of diabetes:

**🩸 Type 1 Diabetes:**
- Usually develops in children/young adults
- Immune system attacks insulin-producing cells
- Requires daily insulin injections
- About 5-10% of diabetes cases

**🩸 Type 2 Diabetes:**
- Most common type (90-95% of cases)
- Body becomes resistant to insulin
- Often develops in adults over 45
- Can sometimes be managed with lifestyle changes

**🩸 Gestational Diabetes:**
- Develops during pregnancy
- Usually goes away after delivery
- Increases risk of Type 2 diabetes later

**Other types:** MODY, LADA, and diabetes caused by other conditions."""
            ],
            "quick_replies": ["Type 1 details", "Type 2 details", "Risk factors", "Prevention"]
        },
        {
            "tag": "symptoms",
            "patterns": [
                "symptoms of diabetes", "signs of diabetes", "diabetes symptoms", "how do i know if i have diabetes",
                "warning signs", "diabetes signs", "early symptoms", "diabetes indicators"
            ],
            "responses": [
                """Common diabetes symptoms include:

**🚨 Classic Symptoms:**
• **Frequent urination** (especially at night)
• **Excessive thirst** that doesn't go away
• **Extreme hunger** even after eating
• **Unexplained weight loss**
• **Fatigue and weakness**

**⚠️ Other Warning Signs:**
• Blurred vision
• Slow-healing cuts/bruises
• Frequent infections
• Tingling in hands/feet
• Dry mouth
• Irritability or mood changes

**🔴 Emergency Signs (seek immediate care):**
• Vomiting and can't keep fluids down
• Blood sugar over 250 mg/dL
• Difficulty breathing
• Confusion

*Note: Some people with Type 2 diabetes may have no symptoms initially.*"""
            ],
            "quick_replies": ["When to see doctor", "Blood sugar levels", "Testing", "Risk factors"]
        },
        {
            "tag": "prevention",
            "patterns": [
                "how to prevent diabetes", "prevent diabetes", "diabetes prevention", 
                "can i prevent diabetes", "avoid diabetes", "reduce diabetes risk",
                "prevention tips", "stop diabetes"
            ],
            "responses": [
                """Here are proven ways to prevent Type 2 diabetes:

**🏃‍♀️ Stay Active:**
• At least 150 minutes of moderate activity per week
• Strength training 2+ times per week
• Even small amounts help!

**🥗 Healthy Eating:**
• Choose whole grains over refined
• Eat plenty of vegetables and fruits
• Limit sugary drinks and processed foods
• Control portion sizes

**⚖️ Maintain Healthy Weight:**
• Losing just 5-7% of body weight helps
• Focus on sustainable changes
• BMI between 18.5-24.9 is ideal

**🚭 Don't Smoke:**
• Smoking increases diabetes risk by 30-40%
• Quitting at any age helps

**💤 Get Enough Sleep:**
• 7-9 hours per night
• Poor sleep affects blood sugar

**🩺 Regular Check-ups:**
• Monitor blood pressure and cholesterol
• Get tested if you have risk factors"""
            ],
            "quick_replies": ["Exercise tips", "Diet plan", "Weight loss", "Risk assessment"]
        },
        {
            "tag": "diet_nutrition",
            "patterns": [
                "diabetes diet", "what to eat", "food for diabetes", "diabetic diet",
                "nutrition diabetes", "foods to avoid", "healthy eating diabetes",
                "meal plan", "diet tips", "sugar intake"
            ],
            "responses": [
                """Diabetes-friendly eating guidelines:

**✅ FOODS TO EMBRACE:**
• **Vegetables:** Leafy greens, broccoli, peppers, tomatoes
• **Lean Proteins:** Fish, chicken, beans, tofu, eggs
• **Whole Grains:** Brown rice, quinoa, oats, whole wheat
• **Healthy Fats:** Avocados, nuts, olive oil, seeds
• **Low-fat Dairy:** Plain yogurt, milk, cheese

**🍎 FRUITS (in moderation):**
• Berries, apples, citrus fruits
• Fresh or frozen without added sugar

**❌ LIMIT OR AVOID:**
• Sugary drinks and sodas
• White bread, pasta, rice
• Fried and processed foods
• Sweets and desserts
• High-sodium foods

**💡 EATING TIPS:**
• Use the plate method: ½ vegetables, ¼ lean protein, ¼ whole grains
• Eat at regular times
• Control portions
• Stay hydrated with water"""
            ],
            "quick_replies": ["Meal planning", "Carb counting", "Recipes", "Portion sizes"]
        },
        {
            "tag": "exercise_fitness",
            "patterns": [
                "exercise diabetes", "diabetes workout", "physical activity", "fitness diabetes",
                "best exercise", "workout routine", "how much exercise", "gym diabetes"
            ],
            "responses": [
                """Exercise is crucial for diabetes management:

**🎯 RECOMMENDED ACTIVITY:**
• **150 minutes** of moderate aerobic activity per week
• **2+ days** of strength training per week
• Break it into manageable chunks (30 min, 5 days)

**🚶‍♀️ AEROBIC EXERCISES:**
• Walking, swimming, cycling
• Dancing, hiking, jogging
• Sports like tennis or basketball

**💪 STRENGTH TRAINING:**
• Weight lifting, resistance bands
• Bodyweight exercises (push-ups, squats)
• Helps improve insulin sensitivity

**⚠️ SAFETY TIPS:**
• Check blood sugar before/after exercise
• Carry glucose tablets if needed
• Stay hydrated
• Start slowly and build up
• Wear proper footwear

**📈 BENEFITS:**
• Lowers blood sugar
• Improves insulin sensitivity
• Helps with weight management
• Reduces stress
• Strengthens heart"""
            ],
            "quick_replies": ["Workout plan", "Safety tips", "Blood sugar exercise", "Getting started"]
        },
        {
            "tag": "blood_sugar_levels",
            "patterns": [
                "blood sugar levels", "glucose levels", "normal blood sugar", "blood glucose",
                "a1c levels", "hba1c", "sugar levels", "glucose range", "target levels"
            ],
            "responses": [
                """Understanding blood sugar levels:

**🎯 TARGET RANGES (for most adults with diabetes):**
• **Before meals:** 80-130 mg/dL
• **2 hours after meals:** Less than 180 mg/dL
• **A1C (3-month average):** Less than 7%

**📊 NORMAL RANGES (no diabetes):**
• **Fasting:** Less than 100 mg/dL
• **After meals:** Less than 140 mg/dL
• **A1C:** Less than 5.7%

**⚠️ CONCERNING LEVELS:**
• **Low (Hypoglycemia):** Below 70 mg/dL
• **High (Hyperglycemia):** Above 250 mg/dL

**🩸 A1C INTERPRETATION:**
• Below 5.7% = Normal
• 5.7-6.4% = Prediabetes
• 6.5% or higher = Diabetes

**📝 MONITORING TIPS:**
• Test at consistent times
• Keep a log
• Note patterns with food/activity
• Share results with your doctor"""
            ],
            "quick_replies": ["How to test", "When to test", "A1C explained", "Managing highs/lows"]
        },
        {
            "tag": "complications",
            "patterns": [
                "diabetes complications", "diabetes problems", "long term effects",
                "diabetes damage", "what happens untreated", "serious effects"
            ],
            "responses": [
                """Diabetes complications and prevention:

**🫀 CARDIOVASCULAR:**
• Heart disease and stroke risk
• High blood pressure
• *Prevention: Control blood sugar, exercise, healthy diet*

**👁️ EYE PROBLEMS:**
• Diabetic retinopathy, cataracts, glaucoma
• *Prevention: Annual eye exams, blood sugar control*

**🦶 FOOT COMPLICATIONS:**
• Nerve damage, poor circulation, infections
• *Prevention: Daily foot care, proper shoes*

**🧠 NERVE DAMAGE (Neuropathy):**
• Tingling, pain, numbness in extremities
• *Prevention: Tight blood sugar control*

**🫘 KIDNEY DISEASE:**
• Can lead to kidney failure
• *Prevention: Control blood pressure and blood sugar*

**🦷 DENTAL PROBLEMS:**
• Gum disease, tooth loss
• *Prevention: Good oral hygiene, regular dental care*

**✅ PREVENTION IS KEY:**
• Keep blood sugar in target range
• Regular medical check-ups
• Take medications as prescribed
• Maintain healthy lifestyle"""
            ],
            "quick_replies": ["Prevention tips", "Warning signs", "Regular check-ups", "Foot care"]
        },
        {
            "tag": "medication_treatment",
            "patterns": [
                "diabetes medication", "insulin", "diabetes treatment", "metformin",
                "diabetes medicine", "pills diabetes", "injections", "drug therapy"
            ],
            "responses": [
                """Diabetes treatment options:

**💊 TYPE 2 DIABETES MEDICATIONS:**
• **Metformin:** First-line treatment, reduces glucose production
• **Sulfonylureas:** Stimulate insulin production
• **SGLT2 inhibitors:** Help kidneys remove glucose
• **GLP-1 agonists:** Slow digestion, increase insulin

**💉 INSULIN THERAPY:**
• **Type 1:** Always requires insulin
• **Type 2:** May need insulin over time
• **Types:** Rapid, short, intermediate, long-acting

**🎯 TREATMENT GOALS:**
• A1C less than 7% (for most adults)
• Blood pressure less than 140/90
• Cholesterol management
• Healthy weight

**⚠️ MEDICATION REMINDERS:**
• Take as prescribed
• Don't skip doses
• Monitor for side effects
• Regular doctor visits
• Never stop without consulting doctor

**🌿 LIFESTYLE IS MEDICINE TOO:**
• Diet and exercise are crucial
• Can reduce medication needs
• Sometimes prevent progression"""
            ],
            "quick_replies": ["Insulin basics", "Side effects", "When needed", "Lifestyle vs medication"]
        },
        {
            "tag": "risk_factors",
            "patterns": [
                "diabetes risk factors", "who gets diabetes", "diabetes risk", "am i at risk",
                "family history diabetes", "genetic diabetes", "causes diabetes"
            ],
            "responses": [
                """Diabetes risk factors:

**🧬 UNCHANGEABLE FACTORS:**
• **Age:** 45+ years
• **Family history:** Parent or sibling with diabetes
• **Race/ethnicity:** Higher risk in African American, Hispanic, Native American, Asian American, Pacific Islander
• **Previous gestational diabetes**

**⚖️ LIFESTYLE FACTORS (You can change!):**
• **Overweight/obesity** (BMI ≥25)
• **Physical inactivity**
• **Poor diet** (high processed foods, sugar)
• **Smoking**
• **High stress levels**

**🩺 MEDICAL CONDITIONS:**
• **Prediabetes** (A1C 5.7-6.4%)
• **High blood pressure** (≥140/90)
• **Low HDL cholesterol** (<40 mg/dL men, <50 women)
• **High triglycerides** (≥250 mg/dL)
• **PCOS** (polycystic ovary syndrome)

**🎯 RISK ASSESSMENT:**
• The more factors, the higher your risk
• Many factors are preventable
• Regular screening if high risk
• Early detection allows prevention"""
            ],
            "quick_replies": ["Risk assessment", "Prevention", "Screening tests", "Prediabetes"]
        },
        {
            "tag": "testing_diagnosis",
            "patterns": [
                "diabetes test", "how is diabetes diagnosed", "diabetes screening",
                "blood test diabetes", "glucose test", "a1c test", "when to get tested"
            ],
            "responses": [
                """Diabetes testing and diagnosis:

**🩸 DIAGNOSTIC TESTS:**
• **A1C Test:** Measures 3-month average blood sugar
  - Normal: <5.7% | Prediabetes: 5.7-6.4% | Diabetes: ≥6.5%
• **Fasting Glucose:** No food for 8+ hours
  - Normal: <100 | Prediabetes: 100-125 | Diabetes: ≥126
• **Random Glucose:** Any time of day
  - Diabetes: ≥200 mg/dL with symptoms
• **Oral Glucose Tolerance Test (OGTT):** Drink glucose solution
  - Normal: <140 | Prediabetes: 140-199 | Diabetes: ≥200

**📅 WHO SHOULD BE TESTED:**
• **Everyone 45+** every 3 years
• **Adults <45** if overweight plus risk factors
• **Pregnant women** (24-28 weeks)
• **Anyone with symptoms**

**⚠️ CONFIRMING DIAGNOSIS:**
• Usually requires two abnormal test results
• Or one test if symptoms present
• Testing should be repeated if borderline

**🎯 EARLY DETECTION BENEFITS:**
• Prevent complications
• Start treatment sooner
• Better long-term outcomes"""
            ],
            "quick_replies": ["Test preparation", "Understanding results", "Prediabetes", "Next steps"]
        },
        {
            "tag": "prediabetes",
            "patterns": [
                "prediabetes", "pre diabetes", "borderline diabetes", "early diabetes",
                "insulin resistance", "glucose intolerance", "reversing prediabetes"
            ],
            "responses": [
                """Understanding prediabetes:

**📊 PREDIABETES RANGES:**
• **A1C:** 5.7-6.4%
• **Fasting glucose:** 100-125 mg/dL
• **OGTT (2-hour):** 140-199 mg/dL

**⚠️ WHAT IT MEANS:**
• Blood sugar higher than normal, but not diabetes yet
• **84 million** Americans have prediabetes
• **90%** don't know they have it
• **15-30%** will develop Type 2 diabetes within 5 years without intervention

**✅ THE GOOD NEWS:**
• **Prediabetes is reversible!**
• Lifestyle changes can prevent/delay Type 2 diabetes
• Small changes make big differences

**🎯 PROVEN PREVENTION STRATEGIES:**
• **Lose 5-7%** of body weight
• **Exercise 150 minutes/week** (brisk walking counts!)
• **Eat healthier** (whole grains, vegetables, lean protein)
• **Limit processed foods** and sugary drinks

**📈 SUCCESS RATES:**
• Lifestyle changes reduce diabetes risk by **58%**
• Even more effective than medication
• Benefits last for years"""
            ],
            "quick_replies": ["Prevention plan", "Lifestyle changes", "Monitoring", "Success stories"]
        },
        {
            "tag": "emergency_help",
            "patterns": [
                "diabetes emergency", "emergency", "help", "urgent", "blood sugar too high",
                "blood sugar too low", "diabetic coma", "ketoacidosis", "hypoglycemia"
            ],
            "responses": [
                """🚨 DIABETES EMERGENCIES - WHEN TO SEEK IMMEDIATE HELP:

**🔴 SEVERE LOW BLOOD SUGAR (Hypoglycemia):**
• Blood sugar <70 mg/dL
• Symptoms: Confusion, shakiness, sweating, unconsciousness
• **IMMEDIATE ACTION:** Give glucose tablets/juice if conscious
• **CALL 911** if unconscious or can't swallow

**🔴 SEVERE HIGH BLOOD SUGAR (DKA):**
• Blood sugar >250 mg/dL
• Symptoms: Vomiting, difficulty breathing, fruity breath, confusion
• **CALL 911 IMMEDIATELY**

**🔴 HYPERGLYCEMIC HYPEROSMOLAR SYNDROME:**
• Blood sugar >600 mg/dL
• Severe dehydration, confusion
• **CALL 911 IMMEDIATELY**

**📱 EMERGENCY ACTIONS:**
1. **Call 911** for severe symptoms
2. **Never drive** if experiencing symptoms
3. **Wear medical ID** bracelet
4. **Have emergency contacts** readily available

**⚠️ This is informational only - always seek immediate medical attention for emergencies!**"""
            ],
            "quick_replies": ["When to call 911", "Low blood sugar help", "High blood sugar help", "Emergency kit"]
        },
        {
            "tag": "thanks",
            "patterns": [
                "thank you", "thanks", "appreciate it", "helpful", "great", "awesome", "perfect"
            ],
            "responses": [
                "You're very welcome! I'm here whenever you need diabetes information. Stay healthy! 💙",
                "Happy to help! Remember, knowledge is power when it comes to diabetes management. 💪",
                "Glad I could assist! Don't hesitate to ask if you have more questions about diabetes. 🏥",
                "You're welcome! Take care of yourself and keep up with healthy habits! 🌟"
            ],
            "quick_replies": ["More questions", "Prevention tips", "Diet advice", "Exercise tips"]
        },
        {
            "tag": "goodbye",
            "patterns": [
                "bye", "goodbye", "see you", "talk to you later", "gtg", "have to go", "farewell"
            ],
            "responses": [
                "Goodbye! Remember to take care of your health. I'm here whenever you need diabetes information! 👋",
                "See you later! Keep up with healthy habits and don't hesitate to ask questions anytime! 🌟",
                "Take care! Wishing you good health and remember - small steps lead to big changes! 💙",
                "Goodbye! Stay healthy and remember that managing diabetes is a journey, not a destination! 🏥"
            ],
            "quick_replies": []
        }
    ]
}

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

class EnhancedDiabetesChatbot:
    def __init__(self):
        self.intents = ENHANCED_INTENTS
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(self, text):
        """Clean and preprocess user input"""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        # Tokenize and lemmatize
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)
    
    def calculate_similarity(self, user_input, pattern):
        """Calculate similarity between user input and pattern"""
        user_words = set(user_input.split())
        pattern_words = set(pattern.split())
        
        if not user_words or not pattern_words:
            return 0
        
        # Jaccard similarity
        intersection = user_words.intersection(pattern_words)
        union = user_words.union(pattern_words)
        
        jaccard = len(intersection) / len(union) if union else 0
        
        # Bonus for substring matches
        substring_bonus = 0
        for word in user_words:
            if any(word in pattern_word or pattern_word in word for pattern_word in pattern_words):
                substring_bonus += 0.1
        
        return jaccard + min(substring_bonus, 0.3)
    
    def find_best_intent(self, user_input):
        """Find the best matching intent based on patterns"""
        preprocessed_input = self.preprocess_text(user_input)
        best_match = None
        best_score = 0
        
        for intent in self.intents["intents"]:
            for pattern in intent["patterns"]:
                score = self.calculate_similarity(preprocessed_input, pattern)
                
                # Direct substring matching bonus
                if pattern in user_input.lower() or user_input.lower() in pattern:
                    score += 0.2
                
                if score > best_score:
                    best_score = score
                    best_match = intent
        
        # Threshold for accepting a match
        if best_score > 0.1:
            return best_match, best_score
        
        return None, 0
    
    def get_response(self, user_input):
        """Generate response based on user input"""
        intent, confidence = self.find_best_intent(user_input)
        
        if intent:
            response = random.choice(intent["responses"])
            quick_replies = intent.get("quick_replies", [])
            return response, quick_replies, intent["tag"]
        else:
            # Fallback responses with helpful suggestions
            fallback_responses = [
                "I'm specialized in diabetes information. Could you ask about symptoms, prevention, diet, exercise, or treatment? 🤔",
                "I didn't quite understand that. Try asking about diabetes types, blood sugar levels, or risk factors! 💭",
                "I'm here to help with diabetes questions! You can ask about complications, medications, or testing. 🏥",
                "Let me help you with diabetes information. Try asking about nutrition, exercise, or management tips! 💪"
            ]
            
            suggested_questions = [
                "What is diabetes?", "Symptoms", "Prevention tips", "Diet advice"
            ]
            
            return random.choice(fallback_responses), suggested_questions, "fallback"

# Initialize the enhanced chatbot
@st.cache_resource
def load_enhanced_chatbot():
    return EnhancedDiabetesChatbot()

def display_quick_replies(quick_replies):
    """Display quick reply buttons"""
    if quick_replies:
        st.markdown("**Quick replies:**")
        cols = st.columns(min(len(quick_replies), 4))
        for i, reply in enumerate(quick_replies[:4]):
            with cols[i % 4]:
                if st.button(reply, key=f"quick_reply_{i}_{reply}"):
                    st.session_state.selected_quick_reply = reply
                    st.rerun()

# Main app
def main():
    st.title("🏥 Enhanced Diabetes Prediction & Information App")
    
    tab1, tab2 = st.tabs(["Diabetes Prediction", "Enhanced AI Diabetes Assistant"])
    
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
        st.markdown("### 🤖 Enhanced AI Diabetes Assistant")
        st.caption("Ask me detailed questions about diabetes - I have comprehensive knowledge about symptoms, prevention, diet, exercise, complications, and more!")
        
        # Load enhanced chatbot
        chatbot = load_enhanced_chatbot()
        
        # Initialize chat history with enhanced welcome message
        if "enhanced_chat_messages" not in st.session_state:
            st.session_state.enhanced_chat_messages = [
                {
                    "role": "assistant", 
                    "content": """👋 Hello! I'm your Enhanced Diabetes AI Assistant. 

I have comprehensive knowledge about:
• 🩺 Diabetes types, symptoms, and diagnosis  
• 🥗 Nutrition and meal planning
• 🏃‍♀️ Exercise and lifestyle management
• 💊 Medications and treatments
• ⚠️ Complications and prevention
• 🩸 Blood sugar monitoring and testing

What would you like to learn about diabetes today?""",
                    "quick_replies": ["What is diabetes?", "Symptoms", "Prevention tips", "Diet advice"],
                    "intent": "greeting"
                }
            ]
        
        # Check for quick reply selection
        if "selected_quick_reply" in st.session_state:
            user_message = st.session_state.selected_quick_reply
            st.session_state.enhanced_chat_messages.append({"role": "user", "content": user_message})
            
            # Get bot response
            with st.spinner("Thinking..."):
                response, quick_replies, intent = chatbot.get_response(user_message)
            
            st.session_state.enhanced_chat_messages.append({
                "role": "assistant", 
                "content": response,
                "quick_replies": quick_replies,
                "intent": intent
            })
            
            del st.session_state.selected_quick_reply
            st.rerun()
        
        # Display chat messages with enhanced styling
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.enhanced_chat_messages):
                if message["role"] == "user":
                    st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>', 
                              unsafe_allow_html=True)
                    
                    # Display quick replies for the latest assistant message
                    if i == len(st.session_state.enhanced_chat_messages) - 1 and "quick_replies" in message:
                        if message["quick_replies"]:
                            st.markdown("---")
                            display_quick_replies(message["quick_replies"])
        
        # Chat input
        if prompt := st.chat_input("Ask a detailed question about diabetes...", key="enhanced_chat"):
            # Add user message to chat history
            st.session_state.enhanced_chat_messages.append({"role": "user", "content": prompt})
            
            # Get bot response
            with st.spinner("Analyzing your question..."):
                response, quick_replies, intent = chatbot.get_response(prompt)
            
            # Add assistant response to chat history
            st.session_state.enhanced_chat_messages.append({
                "role": "assistant", 
                "content": response,
                "quick_replies": quick_replies,
                "intent": intent
            })
            
            st.rerun()
        
        # Enhanced sidebar info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🤖 Enhanced AI Assistant")
        st.sidebar.success("✅ Advanced pattern matching active")
        st.sidebar.info("✅ Comprehensive diabetes knowledge base")
        st.sidebar.info("✅ Smart intent recognition")
        st.sidebar.info("✅ Interactive quick replies")
        
        # Display conversation stats
        if len(st.session_state.enhanced_chat_messages) > 1:
            user_messages = [msg for msg in st.session_state.enhanced_chat_messages if msg["role"] == "user"]
            st.sidebar.markdown(f"**Conversation Stats:**")
            st.sidebar.markdown(f"• Questions asked: {len(user_messages)}")
            
            # Show most discussed topics
            intents = [msg.get("intent", "unknown") for msg in st.session_state.enhanced_chat_messages if msg["role"] == "assistant"]
            if intents:
                from collections import Counter
                top_topics = Counter(intents).most_common(3)
                st.sidebar.markdown("**Top topics discussed:**")
                for topic, count in top_topics:
                    if topic != "unknown" and topic != "fallback":
                        st.sidebar.markdown(f"• {topic.replace('_', ' ').title()}: {count}")
        
        # Enhanced example questions
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 💡 Try asking about:")
        
        example_questions = {
            "🩺 **Basics**": [
                "What are the types of diabetes?",
                "What are early symptoms?",
                "How is diabetes diagnosed?"
            ],
            "🥗 **Nutrition**": [
                "What foods should I avoid?",
                "Can you suggest a meal plan?",
                "How do carbs affect blood sugar?"
            ],
            "🏃‍♀️ **Exercise**": [
                "Best exercises for diabetes?",
                "How much should I exercise?",
                "Exercise safety tips?"
            ],
            "⚠️ **Management**": [
                "What are target blood sugar levels?",
                "Diabetes complications?",
                "When should I see a doctor?"
            ]
        }
        
        for category, questions in example_questions.items():
            st.sidebar.markdown(category)
            for q in questions:
                st.sidebar.markdown(f"• {q}")
        
        # Clear chat button
        st.sidebar.markdown("---")
        if st.sidebar.button("🗑️ Clear Conversation", type="secondary"):
            st.session_state.enhanced_chat_messages = [
                {
                    "role": "assistant", 
                    "content": "Conversation cleared! How can I help you with diabetes information?",
                    "quick_replies": ["What is diabetes?", "Symptoms", "Prevention tips", "Diet advice"],
                    "intent": "greeting"
                }
            ]
            st.rerun()

    # Enhanced app information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 App Information")
    st.sidebar.info("""
    **🔹 Diabetes Prediction:**
    Uses Random Forest Classifier with health metrics including demographics, 
    medical history, lifestyle factors, and blood test results.
    
    **🔹 Enhanced AI Assistant:**
    • Advanced pattern matching algorithm
    • Comprehensive diabetes knowledge base
    • Smart intent recognition
    • Interactive quick reply system
    • 12+ specialized topics covered
    
    **⚠️ Important:** This is an educational tool, not medical advice. 
    Always consult healthcare professionals for diagnosis and treatment.
    """)
    
    # Add usage tips
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Usage Tips")
    st.sidebar.markdown("""
    **For better responses:**
    • Be specific in your questions
    • Use medical terms when known
    • Ask follow-up questions
    • Use the quick reply buttons
    • Try different phrasings if needed
    """)

if __name__ == "__main__":
    main()