import hickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
from preprocessing import preprocess_text

X_train, X_test, y_train, y_test, vocab_size, embedding_matrix, maxlen, tokenizer = hickle.load("preprocessed.hickle")


test_texts = [
    "fucking shit movie so bad omg why is it this shit dont like",
    "i really liked the style of the movie, moreover the style is really fresh and nice"
]

for i, text in enumerate(test_texts):
    test_texts[i] = preprocess_text(text)

tokenized = tokenizer.texts_to_sequences(test_texts)
tokenized = pad_sequences(tokenized, padding='post', maxlen=maxlen)

model = load_model("model.h5")
model.summary()
predict = model.predict(tokenized)

print(predict)
