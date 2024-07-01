import joblib
import sys

def load_model():
    model = joblib.load('spam_detector_model.pkl')
    vectorizer = joblib.load('spam_detector_vectorizer.pkl')
    return model, vectorizer

def predict(text):
    model, vectorizer = load_model()
    transformation = vectorizer.transform([text])
    prediction = model.predict(transformation)
    return "Spam" if prediction[0] == 1 else "Not Spam"

if __name__ == '__main__':
    text = sys.argv[1]
    prediction = predict(text)
    print(f"The Message is: {prediction}")