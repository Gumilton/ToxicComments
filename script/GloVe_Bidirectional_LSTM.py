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


def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')


embed_size = 100
max_features = 20000
maxlen = 100
trainEmbed = True

embeddings_index = dict(get_coefs(*o.strip().split()) for o in open("../data/glove.6B."+str(embed_size)+"d.txt", 'rb'))
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
print(emb_mean,emb_std)

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

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


word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


def get_model(embed_size=embed_size, state_size=50, drop_rate=0.2, lr=0.01):
    # embed_size = 128
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=trainEmbed)(inp)
    x = Bidirectional(LSTM(state_size, return_sequences=True, dropout=drop_rate,
                           recurrent_dropout=drop_rate))(x)
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
isSubmit = False


if isSubmit:
    file_path = "../model/GloVe_fullTrain_weights_base_binary_crossentropy_trainEmbed_" + str(trainEmbed) + "_embedSize_" +str(embed_size) + ".best.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    validation_ratio = 0
    callbacks_list = [checkpoint]
    epochs = 2

else:
    file_path = "../model/GloVe_weights_base_binary_crossentropy_trainEmbed_" + str(trainEmbed) + "_embedSize_" +str(embed_size) + ".best.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='min',
                                 epsilon=0.01, cooldown=0, min_lr=1e-6)
    validation_ratio = 0.1
    callbacks_list = [checkpoint, early, reduceLR] #early
    epochs = 20

hist = model.fit(X_tr, y, batch_size=batch_size, epochs=epochs, validation_split=validation_ratio,
                 callbacks=callbacks_list, verbose=2)

model.load_weights(file_path)

y_test = model.predict(X_te, batch_size=batch_size)

sample_submission = pd.read_csv("../data/sample_submission.csv")

sample_submission[list_classes] = y_test

sample_submission.to_csv("../result/GloVe_binary_crossentropy_trainEmbed_" + str(trainEmbed) + "_embedSize_" +str(embed_size) + ".csv", index=False)
