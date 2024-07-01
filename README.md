Details About this Project:

This Github Repository contains the code for Classification of SMS as Spam and Ham ( Not Spam ) Respectively.

A Model is developed using Multinomial Naive Bayes with an overall accracy of 96 percent. This Model is trained on a Dataset which consists of SMS marked as Spam and HAM.

Steps to run:
1. Docker build -t spam-detector .
2. Docker run --rm spam-detector
3. Docker run --rm spam-detector python predict.py "Enter text".
