import streamlit as st
import pickle
import voyageai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv("VOYAGEAI_API_KEY")

# Check if the API key is loaded properly
if not api_key:
    st.error("API key is not found. Please make sure to set the VOYAGEAI_API_KEY in your .env file.")

# Load the trained SVM model
model_file = 'svm_classifier_model.pkl'
try:
    with open(model_file, 'rb') as file:
        loaded_model = pickle.load(file)
except Exception as e:
    st.error(f"Failed to load the model: {str(e)}")

# Initialize voyageai client
vo = voyageai.Client(api_key=api_key)

def predict(text):
    """Predict the class of input text using the trained model."""
    try:
        # Generate embedding for input text
        embedding = vo.embed(
            [text], model="voyage-3", input_type="document"
        ).embeddings[0]

        # Make prediction
        prediction = loaded_model.predict([embedding])[0]
        confidence = loaded_model.predict_proba([embedding]).max()

        return f"Prediction: {prediction} (Confidence: {confidence:.2f})"
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit app layout
st.title("Text Classification with Trained SVM Model")
st.write("Classify your text using the pre-trained SVM model.")

# Text input from the user
input_text = st.text_area("Enter your text here:", placeholder="Type your text here...", height=200)

# Button to trigger prediction
if st.button("Classify Text"):
    if input_text:
        result = predict(input_text)
        st.success("Prediction complete!")
        st.write(result)
    else:
        st.warning("Please enter some text to classify.")

# Additional information section
st.sidebar.header("About This App")
st.sidebar.write(
    "This app uses a trained SVM model to classify input text."
    " It retrieves predictions and confidence scores from the backend model."
)
