import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout, LSTM
from keras.optimizers import SGD
import pandas as pd
import numpy as np

# reference: https://towardsdatascience.com/an-easy-tutorial-about-sentiment-analysis-with-deep-learning-and-keras-2bf52b9cba91
# https://medium.com/@gabriel.mayers/sentiment-analysis-from-tweets-using-recurrent-neural-networks-ebf6c202b9d5


def toString(arr):
    return [str(elem) for elem in arr]


def get_data():
    train_data = pd.read_csv("../data/train.csv")
    test_data = pd.read_csv("../data/test.csv")

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

    return x_train, y_train, x_test, y_test, max_words, max_len


def main():
    x_train, y_train, x_test, y_test, max_words, max_len = get_data()
    model = Sequential()
    numNeurons = 512
    model.add(Embedding(max_words, 128, input_length=max_len))
    model.add(Flatten())

    model.add(Dense(numNeurons))
    model.add(Dropout(0.4))
    model.add(Dense(numNeurons))
    model.add(Dropout(0.4))

    model.add(Dense(1, activation='sigmoid'))

    epochs = 5
    batch_size = 128
    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=0.01),
                  metrics=['accuracy'])

    print('Train...')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))

    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

    results = {}
    results['accuracy'] = toString(history.history['accuracy'])
    results['val_accuracy'] = toString(history.history['val_accuracy'])
    results['val_loss'] = toString(history.history['val_loss'])
    results['loss'] = toString(history.history['loss'])

    with open('../results/ffnn_results.json', 'w') as outfile:
        json.dump(results, outfile)


if __name__ == "__main__":
    main()
