import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the model from the saved pickle file
with open('model.pkl', 'rb') as f:
    classifiers, vectorizer = pickle.load(f)

# Define the Streamlit app
def main():
    st.title('SMS Spam Classifier')

    # Get user input
    user_input = st.text_input('Enter a message:')

    # Validate input before making a prediction
    if st.button('Classify'):
        if not user_input or user_input.isspace():  # Check if input is empty or contains only whitespace characters
            st.write("Error: Please enter a non-empty message.")
        else:
            # Preprocess the input text using the loaded vectorizer
            X = vectorizer.transform([user_input])

            # Use the loaded classifiers to make a prediction
            prediction = classifiers[2].predict(X)[0]  # Assuming MultinomialNB is the third classifier
            st.write(f'Prediction: {prediction}')

if __name__ == '__main__':
    main()