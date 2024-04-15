
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle


# Load data
data = pd.read_csv('spam.csv', encoding='latin-1')

column_names = data.columns

# Check for missing values initially
missing_values = data.isnull().sum()
#print(missing_values)

# Check for duplicates
duplicates = data.duplicated().sum()
#print(f"Number of duplicate rows: {duplicates}")

# Remove duplicates
data.drop_duplicates(inplace=True)

# Impute missing values in categorical columns with mode
data.fillna(data.mode().iloc[0], inplace=True)

#After Cleaning, there are no missing values
data.isnull().sum()

#After cleaning there are no duplicates
data.duplicated().sum()



def classify_ensemble(data, vectorizer, classifiers, label_column='v1', text_column='v2', test_size=0.2, random_state=None):
    try:
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(data[text_column], data[label_column], test_size=test_size, random_state=random_state)

        # Preprocess text data using the provided vectorizer
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Initialize lists to store train and test accuracies for each model
        train_accuracies = []
        test_accuracies = []

        # Train each classifier and evaluate
        for classifier in classifiers:
            classifier.fit(X_train_vec, y_train)
            y_train_pred = classifier.predict(X_train_vec)
            y_test_pred = classifier.predict(X_test_vec)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

            print(f"{classifier.__class__.__name__} - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Create ensemble of classifiers
        ensemble = VotingClassifier(estimators=[(f'classifier_{i}', clf) for i, clf in enumerate(classifiers)])
        ensemble.fit(X_train_vec, y_train)

        # Evaluate ensemble
        ensemble_score = ensemble.score(X_test_vec, y_test)
        print(f"Ensemble Accuracy: {ensemble_score:.4f}")

        return train_accuracies, test_accuracies, ensemble_score

    except Exception as e:
        print(f"An error occurred: {e}")


classifiers = [
    RandomForestClassifier(n_estimators=400, n_jobs=1, max_depth=20),
    DecisionTreeClassifier(max_depth=10),
    MultinomialNB(alpha=0.5, fit_prior=True)
]

# Instantiate TfidfVectorizer

vectorizer = TfidfVectorizer()

if __name__ == '__main__':
    # Load data
    data = pd.read_csv('spam.csv', encoding='latin-1')
    classify_ensemble(data, vectorizer, classifiers)

with open('model.pkl', 'wb') as f:
    pickle.dump((classifiers, vectorizer), f)
