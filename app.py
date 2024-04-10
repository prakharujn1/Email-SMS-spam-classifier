import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Porter Stemmer
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Set page title and description
st.set_page_config(page_title="Spam Classifier", page_icon=":email:", layout="wide")

# Custom CSS for background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main title with custom styling
st.markdown(
    """
    <h1 style='text-align: center; color: #0077b6;'>Email/SMS Spam Classifier</h1>
    """,
    unsafe_allow_html=True
)

# Text input area for user to enter message
input_sms = st.text_area("Enter the message")

# Button to trigger prediction
if st.button('Predict'):
    # Preprocess text
    transformed_sms = transform_text(input_sms)
    # Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # Predict
    result = model.predict(vector_input)[0]
    # Display result
    if result == 1:
        st.error("Spam")
    else:
        st.success("Not Spam")
