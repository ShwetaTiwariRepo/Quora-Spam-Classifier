# Quora-Spam-Classifier
Given the dump of Quora questions and spam flag, built a model to predict if a question on Quora is spam or not

# Data Set : 
https://www.dropbox.com/sh/kpf9z73woodfssv/AAAw1_JIzpuVvwteJCma0xMla?dl=0

# Programming Language : 
Python using nltk & Keras

# Model Architecture : 
Deep Learning using Long Short-Term Memory (LSTM) networks

## About Data Set

Dataset has dump of question asked on Quora with a question id with target value of 0 or 1 where 1 identified it as spam.

## Data Pre-Processing

1. Read train.csv File
2. Assign question text to dataframe X 
3. Assign target to dataframe y 
4. Use NLTK to tokenize the word find the longest sentance
5. Get the max number of words most of questions(0.9999%) data contains
6. Convert y to numpy array using to_categorical 
7. Split dataset to train and test for test size of 0.2

![image](https://user-images.githubusercontent.com/64772772/96828884-0c3d0280-1456-11eb-9373-49e450f38d67.png)

## Model Building 

1. constructs the model with 128 LSTM units and with embeding size of 100

![image](https://user-images.githubusercontent.com/64772772/96829248-d9473e80-1456-11eb-8434-41a05e555857.png)

2. Train the model
![image](https://user-images.githubusercontent.com/64772772/96829433-48bd2e00-1457-11eb-8dff-087930713675.png)


## Check Model Performance/Accuracy

### Evaluate the model
![image](https://user-images.githubusercontent.com/64772772/96829529-7bffbd00-1457-11eb-8d59-f2297912a043.png)


### Get predection on new text

![image](https://user-images.githubusercontent.com/64772772/96829746-da2ca000-1457-11eb-802a-fdca467d75e9.png)

## Save Model with Weights
Saving the model for transfer learning or model execution later

![image](https://user-images.githubusercontent.com/64772772/96829962-3263a200-1458-11eb-888c-6323d0bf3fac.png)
