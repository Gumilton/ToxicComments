# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

import numpy as np
import pandas as pd
# Any results you write to the current directory are saved as output.

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.optimizers import Adam
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

max_features = 20000
maxlen = 100

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
# train_ind = np.random.randInt()
train = train.sample(frac=1)

list_sentences_train = train["comment_text"].fillna("gumilton").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("gumilton").values


tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_tr = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)



def get_model(embed_size=128, state_size=50, drop_rate=0.15, lr=0.005):
    # embed_size = 128
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(state_size, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    # x = Dropout(drop_rate)(x)
    x = Dense(state_size, activation="relu")(x)
    x = Dropout(drop_rate)(x)
    x = Dense(state_size, activation="relu")(x)
    # x = Dropout(drop_rate)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy'])

    return model


model = get_model()
batch_size = 1024
isSubmit = True


if isSubmit:
    file_path="fullTrain_weights_base_binary_crossentropy.best.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    validation_ratio = 0
    callbacks_list = [checkpoint]
    epochs = 2

else:
    file_path="weights_base_binary_crossentropy.best.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, mode='min',
                                 epsilon=0.05, cooldown=0, min_lr=10e-6)
    validation_ratio = 0.1
    callbacks_list = [checkpoint, early, reduceLR] #early
    epochs = 20

model.fit(X_tr, y, batch_size=batch_size, epochs=epochs, validation_split=validation_ratio, callbacks=callbacks_list)

model.load_weights(file_path)

y_test = model.predict(X_te, batch_size=batch_size)

sample_submission = pd.read_csv("../data/sample_submission.csv")

sample_submission[list_classes] = y_test

sample_submission.to_csv("binary_crossentropy.csv", index=False)
