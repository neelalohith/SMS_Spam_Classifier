Details About this Project:

This Github Repository contains the code for Classification of SMS as Spam and Ham ( Not Spam ) Respectively.

A Model is developed using Ensemble Learning Technique consisting of Decision Tree, Random Forest And Multinomial Naive Bayes with an overall accracy of 96 percent. This Model is trained on a 'spam.csv' Dataset which consists of SMS marked as Spam and HAM.

Steps Involved In this Project:
1. Data Cleaning
2. Building an Ensemble learning Model along with Vectorization of Input using TF-IDF Technique.
3. Defining The classifier as a set of Ensemble Models.
4. Export the Model as a Pickel file for building a website using streamlit package.
5. Text processing of input using Tokenization is incorporated.
6. Once, Input is entered in the website, the prediction is returned as output.

Steps For running the files:
1. Download the code locally
2. Enter 'streamlit run app.py' in the terminal to runb the app. In the app , Enter the text in a textbox and then the predicted result is returned as output.
3. To run the model and view its accuracy, run 'python3 model.py'.
