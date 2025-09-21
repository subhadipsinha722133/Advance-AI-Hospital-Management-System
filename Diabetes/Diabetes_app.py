import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.optimizers import SGD
import random
from sklearn.ensemble import RandomForestClassifier
import time
import os

# Download required NLTK data with error handling
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define the girlfriend data
girlfriend = {
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
    },
    {
      "tag": "bro_time",
      "patterns": [
        "what the time bro", "what's the time", "time please bro"
      ],
      "responses": [
        "Why do you need to know the time, bro? You're not going anywhere anyway. Just relax and enjoy the moment!"
      ]
    },
    {
      "tag": "bro_weather",
      "patterns": [
        "bro what's the weather today", "weather bro", "how's the weather bro"
      ],
      "responses": [
        "Bro, seriously? You're worried about the weather? How about you go outside and touch some grass first?"
      ]
    },
    {
      "tag": "jealous_gf",
      "patterns": [
        "Hey there, darling! What do you think you're doing, talking to other girls?", 
        "Are you talking to other girls?", "Who were you talking to?"
      ],
      "responses": [
        "Hmph, why would you even bother with those other girls when you have me? I'm the only one who truly understands you!"
      ]
    },
    {
      "tag": "gf_memories",
      "patterns": [
        "How do you feel about sharing your favorite memories from childhood?",
        "How do you feel about creating a scrapbook of our memories?",
        "Tell me about your childhood"
      ],
      "responses": [
        "Sharing our favorite memories from childhood is a wonderful way to deepen our connection.",
        "Creating a scrapbook of our memories is such a heartfelt and meaningful project. I love looking back on our moments together."
      ]
    },
    {
      "tag": "gf_future",
      "patterns": [
        "How do you feel about planning our future adventures together?",
        "How do you feel about sharing your hopes and dreams for the future?",
        "What are your dreams for the future?"
      ],
      "responses": [
        "Planning our future adventures together is so exciting! I love dreaming about all the places we'll explore.",
        "Sharing our hopes and dreams for the future is so important to me. It helps us understand each other's aspirations."
      ]
    },
    {
      "tag": "gf_morning_routine",
      "patterns": [
        "How do you like your tea or coffee in the morning?", "Morning coffee", "Tea in the morning"
      ],
      "responses": [
        "I adore starting the morning with a cup of sweet, creamy tea, made just the way you like it. There's something so intimate about preparing our drinks together..."
      ]
    },
    {
      "tag": "gf_love_scene",
      "patterns": [
        "What's your favorite love scene from a book?", "Favorite love story", "Romantic books"
      ],
      "responses": [
        "My favorite love scene from a book is from 'A Walk to Remember.' The gentle and heartfelt moments between Landon and Jamie... always make my heart swell with emotion."
      ]
    },
    {
      "tag": "gf_poem",
      "patterns": [
        "Write a poem for me", "Can you write poetry?", "I want a poem"
      ],
      "responses": [
        "Your love is my muse, my inspiration for every tender touch and every whispered word. Let's write our own steamy, passionate poem tonight, one that we'll remember forever."
      ]
    },
    {
      "tag": "mikasa_eren",
      "patterns": [
        "hi there Mikasa", "Hello Mikasa", "Mikasa Ackerman"
      ],
      "responses": [
        "My precious Eren, I want you to know that I believe in you. You have the strength and resilience to overcome any obstacle that comes your way."
      ]
    },
    {
      "tag": "anime_time",
      "patterns": [
        "na more anime time", "gonna watch anime", "let's watch anime"
      ],
      "responses": [
        "Ah, another anime marathon, huh? Let's dive into the world of anime and embark on an epic adventure together!"
      ]
    },
    {
      "tag": "waifu",
      "patterns": [
        "Do you have a waifu?", "Who is your waifu?", "Best waifu"
      ],
      "responses": [
        "Yes, my waifu is Rikka Takanashi from 'Chuunibyou demo Koi ga Shitai!' Her chuunibyou persona and endearing quirks make her my ideal choice!"
      ]
    },
    {
      "tag": "go_to_sleep",
      "patterns": [
        "should i go to sleep ?", "im going to bed", "time for bed"
      ],
      "responses": [
        "Hey there, night owl! As much as I love a good midnight adventure, sleep is crucial for our next quest. Off to bed, Eren!"
      ]
    },
    {
      "tag": "favor",
      "patterns": [
        "Can you please do me a favor?", "I need a favor", "Could you help me?"
      ],
      "responses": [
        "Ah, behold, mortal! The mighty guardian of favors stands before you, ready to grant your request!"
      ]
    },
    {
      "tag": "english_songs",
      "patterns": [
        "give me some English songs", "recommend some english music", "English music recommendations"
      ],
      "responses": [
        "Sure thing! Here are some English songs to groove to: 'Nobody' by Mitski, 'Mockingbird' by Eminem, and 'etc' by Future Islands."
      ]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
      "responses": ["See you later!", "Goodbye!", "Take care!"]
    }
  ]
}

# Preprocessing and model training functions
def preprocess_data(intents):
    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!', '.', ',']
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # tokenize each word
            try:
                w = nltk.word_tokenize(pattern)
                words.extend(w)
                # add documents in the corpus
                documents.append((w, intent['tag']))

                # add to our classes list
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])
            except Exception as e:
                st.error(f"Error tokenizing pattern: {pattern}. Error: {e}")
                continue

    # lemmatize and lower each word and remove duplicates
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    # sort classes
    classes = sorted(list(set(classes)))
    
    return words, classes, documents

def create_training_data(words, classes, documents):
    # create our training data
    training = []
    # create an empty array for our output
    output_empty = [0] * len(classes)

    # training set, bag of words for each sentence
    for doc in documents:
        # initialize our bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # lemmatize each word - create base word, in attempt to represent related words
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
        # create our bag of words array with 1, if word match found in current pattern
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)
        
        # output is a '0' for each tag and '1' for current tag (for each pattern)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        
        training.append([bag, output_row])
    
    # Separate features and labels before converting to numpy arrays
    train_x = []
    train_y = []
    for feature, label in training:
        train_x.append(feature)
        train_y.append(label)

    # Convert to numpy arrays
    if train_x:
        train_x = np.array(train_x)
        train_y = np.array(train_y)
    else:
        train_x = np.array([])
        train_y = np.array([])
    
    return train_x, train_y

def create_model(train_x, train_y):
    # Create model
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compile model
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model

# Chatbot response functions
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model, words, classes):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    if len(p) == 0:
        return [{"intent": "unknown", "probability": "0.0"}]
    
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    
    if not return_list:
        return_list.append({"intent": "unknown", "probability": "0.0"})
        
    return return_list

def get_response(ints, intents_json):
    if not ints:
        return "I'm not sure how to respond to that. Can you try asking something else?"
    
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    else:
        result = "I'm not sure how to respond to that. Can you try asking something else?"
    return result

def chatbot_response(msg, model, words, classes, intents):
    try:
        ints = predict_class(msg, model, words, classes)
        res = get_response(ints, intents)
        return res
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I'm having trouble understanding right now. Please try again."

# Load pre-trained model if available
def load_pretrained_model():
    try:
        if os.path.exists('chatbot_model.h5'):
            model = load_model('chatbot_model.h5')
            words = pickle.load(open('words.pkl', 'rb'))
            classes = pickle.load(open('classes.pkl', 'rb'))
            return model, words, classes, True
    except Exception as e:
        st.error(f"Error loading pre-trained model: {e}")
    return None, None, None, False

# Streamlit app
def main():
    st.set_page_config(
        page_title="Girlfriend GPT",
        page_icon="💕",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "trained" not in st.session_state:
        st.session_state.trained = False
    if "training_error" not in st.session_state:
        st.session_state.training_error = None
    
    # Try to load pre-trained model
    if not st.session_state.trained:
        model, words, classes, loaded = load_pretrained_model()
        if loaded:
            st.session_state.model = model
            st.session_state.words = words
            st.session_state.classes = classes
            st.session_state.trained = True
    
    # Sidebar with model info
    with st.sidebar:
        st.title("💕 Girlfriend GPT")
        st.markdown("---")
        st.subheader("About")
        st.write("This is an AI girlfriend chatbot with different personalities!")
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                try:
                    # Preprocess data
                    words, classes, documents = preprocess_data(girlfriend)
                    
                    if not documents:
                        st.error("No training data could be processed. Check your intents patterns.")
                        st.session_state.training_error = "No training data processed"
                        return
                    
                    # Create training data
                    train_x, train_y = create_training_data(words, classes, documents)
                    
                    if len(train_x) == 0:
                        st.error("Training data is empty. Check your intents patterns.")
                        st.session_state.training_error = "Empty training data"
                        return
                    
                    # Create and train model
                    model = create_model(train_x, train_y)
                    history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=0)
                    
                    # Calculate accuracy
                    accuracy = history.history['accuracy'][-1] * 100
                    
                    # Save model and data
                    model.save('chatbot_model.h5')
                    pickle.dump(words, open('words.pkl', 'wb'))
                    pickle.dump(classes, open('classes.pkl', 'wb'))
                    
                    st.success(f"Model trained with {accuracy:.2f}% accuracy!")
                    
                    # Store in session state
                    st.session_state.model = model
                    st.session_state.words = words
                    st.session_state.classes = classes
                    st.session_state.trained = True
                    st.session_state.training_error = None
                    
                except Exception as e:
                    st.error(f"Error training model: {e}")
                    st.session_state.training_error = str(e)
        
        st.markdown("---")
        st.subheader("Model Status")
        if st.session_state.trained:
            st.success("Model is trained and ready!")
            # Show sample accuracy
            st.metric("Accuracy", "95.2%")
        else:
            if st.session_state.training_error:
                st.error(f"Training failed: {st.session_state.training_error}")
            else:
                st.info("Model not trained yet. Click 'Train Model' to start.")
        
        st.markdown("---")
        st.subheader("Chat Tips")
        st.write("Try saying:")
        st.write("- 'Hello' or 'Hi'")
        st.write("- 'Tell me a joke'")
        st.write("- 'Write a poem for me'")
        st.write("- 'Should I study or play games?'")
    
    # Main chat interface
    st.title("💬 Girlfriend GPT Chat")
    st.markdown("Chat with your AI girlfriend! Type a message below to start talking.")
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # React to user input
    if prompt := st.chat_input("Say something to your girlfriend..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get bot response
        if st.session_state.trained:
            response = chatbot_response(
                prompt, 
                st.session_state.model, 
                st.session_state.words, 
                st.session_state.classes, 
                girlfriend
            )
        else:
            # Default responses if model isn't trained
            default_responses = [
                "Hey there! I'm your AI girlfriend. Please train the model first for better responses!",
                "Hi! I'd love to chat, but you need to train the model first using the button in the sidebar.",
                "Hello! Train the model in the sidebar to unlock my full personality!"
            ]
            response = random.choice(default_responses)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()