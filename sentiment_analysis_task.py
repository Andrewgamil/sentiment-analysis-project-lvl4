import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.models import Sequential


Words = 15000
Lentgh = 500
Embed_Size = 32

# Loading Dataset
(train_imgs, train_lables), (test_imgs, test_lables) = imdb.load_data(num_words=Words)

# Padding
trainig_data = sequence.pad_sequences(train_imgs, maxlen=Lentgh)
testng_data = sequence.pad_sequences(test_imgs, maxlen=Lentgh)

# Building Model
modl = Sequential()
modl.add(Embedding(Words, Embed_Size, input_length=Lentgh))
modl.add(SimpleRNN(32))
modl.add(Dense(1, activation='sigmoid'))

# Compileing Model
modl.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Training Model
fit_modl = modl.fit(trainig_data, train_lables,
                    batch_size=128,
                    epochs=10,
                    validation_split=0.2)

# Evaluating Model
evl = modl.evaluate(testng_data, test_lables)
print("Test Loss:", evl[0], "Test Accuracy:", evl[1])



git filter-branch --env-filter '
OLD_EMAIL="andrewgamil18@gmail.com"

if [ "$GIT_COMMITTER_EMAIL" = "$OLD_EMAIL" ]; then

    export GIT_COMMITTER_DATE="2025-03-22T22:00:00+02:00"
fi
if [ "$GIT_AUTHOR_EMAIL" = "$OLD_EMAIL" ]; then

    export GIT_AUTHOR_DATE="2025-03-22T22:00:00+02:00"
fi
' -- --all


