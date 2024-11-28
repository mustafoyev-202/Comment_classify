import streamlit as st
import pickle
import voyageai

# Load the trained SVM model
model_file = 'svm_classifier_model.pkl'
try:
    with open(model_file, 'rb') as file:
        loaded_model = pickle.load(file)
except Exception as e:
    st.error(f"Failed to load the model: {str(e)}")

# Initialize voyageai client
vo = voyageai.Client(api_key="pa-4C6Ct4Qal2uerB7lwo-0cSDSvaYK9LuFEE5Wco4BL7k")


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
