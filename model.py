import hickle
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Activation, Dropout, Dense, Flatten, GlobalMaxPooling1D, Embedding
import numpy as np

X_train, X_test, y_train, y_test, vocab_size, embedding_matrix, maxlen, _ = hickle.load("preprocessed.hickle")


model = Sequential()
model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False))

model.add(LSTM(128))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())

history = model.fit(X_train, np.array(y_train), batch_size=128, epochs=13, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, np.array(y_test), verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])

model.save("model.h5")
