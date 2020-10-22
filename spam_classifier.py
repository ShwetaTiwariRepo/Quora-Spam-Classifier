from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import pandas as pd



# load the data
train=pd.read_csv("train.csv")


X=train["question_text"]
y=train["target"]

#get the max length of the vocubalry 
from nltk import word_tokenize
sent_lens_t=[]
for sent in train["question_text"]:
    sent_lens_t.append(len(word_tokenize(sent)))
max(sent_lens_t)

#Get maximum word length most of sentences have
np.quantile(sent_lens_t,0.9999)

#define the padding for sentences length less than 70, a little more than 64
SEQUENCE_LENGTH=70

# Text tokenization
# vectorizing text, turning each text into sequence of integers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
# lets dump it to a file, so we can use it in testing
pickle.dump(tokenizer, open("Results/tokenizer.pickle", "wb"))

# convert to sequence of integers
X = tokenizer.texts_to_sequences(X)

# pad sequences at the beginning of each sequence with 0's
X = pad_sequences(X, maxlen=SEQUENCE_LENGTH)

y=to_categorical(y)

TEST_SIZE = 0.20 #ratio of testing set
# split and shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=7)


import tqdm
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.models import Sequential
import keras_metrics

def get_embedding_vectors(tokenizer, dim=100):
    embedding_index = {}
    with open(f"data/glove.6B.{dim}d.txt", encoding='utf8') as f:
        for line in tqdm.tqdm(f, "Reading GloVe"):
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vectors

    word_index = tokenizer.word_index
    # we do +1 because Tokenizer() starts from 1
    embedding_matrix = np.zeros((len(word_index)+1, dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found will be 0s
            embedding_matrix[i] = embedding_vector
            
    return embedding_matrix


# constructs the model with 128 LSTM units and with embeding size of 100
lstm_units=128
EMBEDDING_SIZE=100
# get the GloVe embedding vectors
embedding_matrix = get_embedding_vectors(tokenizer)
model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1,
          EMBEDDING_SIZE,
          weights=[embedding_matrix],
          trainable=False,
          input_length=SEQUENCE_LENGTH))

model.add(LSTM(lstm_units, recurrent_dropout=0.2))
model.add(Dropout(0.3))
model.add(Dense(2, activation="softmax"))
# compile as rmsprop optimizer
# aswell as with recall metric
model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
              metrics=["accuracy", keras_metrics.precision(), keras_metrics.recall()])
model.summary()



# print our data shapes
print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)
# train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          batch_size=100, epochs=5,
          #callbacks=[tensorboard, model_checkpoint],
          verbose=1)

# get the loss and metrics
result = model.evaluate(X_test, y_test)
# extract those
loss = result[0]
accuracy = result[1]
precision = result[2]
recall = result[3]

# Check for prediction result
predVal=model.predict(X_test)


# Convert question from number to text
X_test_N = tokenizer.sequences_to_texts(X_test)

#
X_test_N=pd.DataFrame(X_test_N)


Pred_One = pd.DataFrame(data = predVal,columns = ["zero","One"])

X_test_N["OneProb"]=Pred_One["One"]
X_test_N["ZeroProb"]=Pred_One["zero"]

#total spams predicted
X_test_N[X_test_N["OneProb"]>=0.50]

X_test_N.shape


print(f"[+] loss:   {loss*100:.2f}%",\
      f"[+] Accuracy: {accuracy*100:.2f}%",\
          f"[+] Precision:   {precision*100:.2f}%",\
              f"[+] Recall:   {recall*100:.2f}%")
    
int2label = {0: "ham", 1: "spam"}

def get_predictions(text):
    sequence = tokenizer.texts_to_sequences([text])
    # pad the sequence
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    # get the prediction
    prediction = model.predict(sequence)[0]
    # one-hot encoded vector, revert using np.argmax
    return int2label[np.argmax(prediction)]
    #return prediction


text = input("Enter the text to classify:")
print(get_predictions(text))
    
# Save model weights
model.save_weights("spam_classifier.h5")


    