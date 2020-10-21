# Quora-Spam-Classifier
Given the dump of Quora questions and spam flag, built a model to predict if a question on Quora is spam or not

# Data Set : https://www.dropbox.com/sh/kpf9z73woodfssv/AAAw1_JIzpuVvwteJCma0xMla?dl=0

# Programming Language : Python using nltk & Keras

# Model Architecture : Deep Learning using Long Short-Term Memory (LSTM) networks

# About Data Set

Dataset has dump of question asked on Quora with a question id with target value of 0 or 1 where 1 identified it as spam.

# Data Pre-Processing

1. Read train.csv File
2. Assign question text to dataframe X 
3. Assign target to dataframe y 
4. Use NLTK to tokenize the word find the longest sentance
5. Get the max number of words most of questions(0.9999%) data contains
6. Convert y to numpy array using to_categorical 

7. Split dataset to train and test for test size of 0.2

# Model Building 

8. Load embedding vectors from Glove of 100 dimensions 

9. constructs the model with 128 LSTM units and with embeding size of 100

10. Train the model

# Check Model Performance/Accuracy

11. Evaluate the model

12. Check the prediction on test dataset

# Make Random Validation

13. Get predection on new text

# Save Model with Weights
Saving the model for transfer learning or model execution later
