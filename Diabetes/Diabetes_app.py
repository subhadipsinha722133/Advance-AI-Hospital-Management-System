import streamlit as st
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
        page_title="Girlfriend GPT",
        page_icon="💕",
        layout="wide"
    )
    
    # Load model
    model, words, classes, intents = load_chatbot_model()
    
    # Check if model loaded properly
    if model is None:
        st.warning("AI model not loaded. Using demo mode with pattern matching.")

    # Sidebar with info
    with st.sidebar:
        st.title("💕 Girlfriend GPT")
        st.sidebar.image("girlfriend.gif")
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
    st.title("💬 Girlfriend GPT Chat")
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

if __name__ == "__main__":

    main()

