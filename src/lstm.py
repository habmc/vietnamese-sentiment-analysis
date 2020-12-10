import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout, LSTM
from keras.optimizers import SGD
import pandas as pd
import numpy as np


def toString(arr):
    return [str(elem) for elem in arr]


path = "../"
train_data = pd.read_csv(path + "data/train.csv")
test_data = pd.read_csv(path + "data/test.csv")

training_sent = train_data['review'].tolist()
training_labels = train_data['label'].tolist()

testing_sent = test_data['review'].tolist()
testing_labels = test_data['label'].tolist()

training_sent = [elem.strip(' "') for elem in training_sent]
testing_sent = [elem.strip(' "') for elem in testing_sent]

max_words = 6000
max_len = max(max([len(elem) for elem in training_sent]),
              max([len(elem) for elem in testing_sent]))

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(training_sent)

training_seq = tokenizer.texts_to_sequences(training_sent)
x_train = sequence.pad_sequences(training_seq, maxlen=max_len)
y_train = np.array(training_labels)

testing_seq = tokenizer.texts_to_sequences(testing_sent)
x_test = sequence.pad_sequences(testing_seq, maxlen=max_len)
y_test = np.array(testing_labels)

model = Sequential()
numNeurons = 256
model.add(Embedding(max_words, 128, input_length=max_len))

model.add(LSTM(numNeurons, return_sequences=True))
model.add(LSTM(numNeurons, return_sequences=True))
model.add(LSTM(numNeurons))

model.add(Dense(1, activation='sigmoid'))

epochs = 20
batch_size = 128
model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=0.05),
              metrics=['accuracy'])

print('Train...')
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.33)

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

results = {}
results['accuracy'] = toString(history.history['accuracy'])
results['val_accuracy'] = toString(history.history['val_accuracy'])
results['val_loss'] = toString(history.history['val_loss'])
results['loss'] = toString(history.history['loss'])

with open(path + 'results/lstm_results.json', 'w') as outfile:
    json.dump(results, outfile)

model.save(path + 'weights/lstm.h5')
