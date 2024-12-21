# Sarcasm Detection with TF-IDF and Streamlit

This project focuses on detecting sarcasm in textual data using a machine learning model trained with **TF-IDF features**. A user-friendly interface is built using **Streamlit**, allowing users to test the sarcasm detection model interactively.

## Features

- Preprocesses text using TF-IDF vectorization.
- Trains a sarcasm detection model using Logistic Regression.
- Provides a Streamlit-based web interface for testing the model.

## Tech Stack
- Programming Language: Python
- Libraries:
  - Machine Learning: scikit-learn
  - NLP: nltk
  - Web Framework: streamlit
    
# Getting Started
## Prerequisites
- Python 3.7+
- Install required libraries:
```bash
pip install -r requirement.txt
```

## Training the Model
- Open model_training.ipynb in Jupyter Notebook.
- Follow the step-by-step instructions to:
    - Load the dataset.
    - Preprocess the text using TF-IDF Vectorizer.
    - Train the Logistic Regression model.
    - Save the trained model and vectorizer using joblib.
 
## Running the Web App
- Open a terminal and navigate to the project directory.
- Run the Streamlit app using:
```bash
streamlit run app.py
```
- Open the URL displayed in the terminal (e.g., http://localhost:8501).
- Input text to test for sarcasm.

## How It Works
- Data Preprocessing:
   - Tokenization, stopword removal, and TF-IDF vectorization are applied to transform text into numerical features.
- Model Training:
   - Logistic Regression is used to train the model on the processed data.
- Web Interface:
   - Users input text via the Streamlit app.
   - The model predicts whether the input text is sarcastic or not.

  

  

