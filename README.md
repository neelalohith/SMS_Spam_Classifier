Details About this Project:

This Github Repository contains the code for Classification of SMS as Spam and Ham ( Not Spam ) Respectively.

A Model is developed using Multinomial Naive Bayes with an overall accracy of 96 percent. This Model is trained on a Dataset which consists of SMS marked as Spam and HAM.

Steps to run:
1. Docker build -t spam-detector .
2. Docker run --rm spam-detector
3. Docker run --rm spam-detector python predict.py "Enter text".

Results:

<img width="600" alt="Screenshot 2024-07-01 at 10 32 50 PM" src="https://github.com/neelalohith/SMS_Spam_Classifier/assets/98219059/78d2e6b2-3803-4210-9eca-1e66390e0729">
<br />
<img width="841" alt="Screenshot 2024-07-01 at 10 34 06 PM" src="https://github.com/neelalohith/SMS_Spam_Classifier/assets/98219059/7ec6b9da-8bbd-4fb2-bdd1-654d2addd4e8">
<br />
Note: The above latter image was run locally, we can run in docker as well.

