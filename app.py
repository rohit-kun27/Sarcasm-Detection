import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
model = joblib.load("sarcasm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit App
st.title("Sarcasm Detection Web App")
st.write("Enter a sentence below to check if it's sarcastic or not.")

# Input text from user
user_input = st.text_input("Enter a sentence:")

if st.button("Check Sarcasm"):
    if user_input:
        # Preprocess and vectorize the input
        input_tfidf = vectorizer.transform([user_input.lower()])
        
        # Predict using the model
        prediction = model.predict(input_tfidf)[0]
        
        # Output result
        if prediction == 1:
            st.success("This sentence is **Sarcastic**! 🤔")
        else:
            st.info("This sentence is **Not Sarcastic**.")
    else:
        st.warning("Please enter a sentence to analyze.")

# Footer
st.write("---")
st.write("Built with ❤️ using Streamlit")
