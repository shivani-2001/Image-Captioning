import ast
import numpy as np
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Input, Dropout, Dense, Embedding, LSTM, add

encode_model = InceptionV3(weights='imagenet')
encode_model = Model(encode_model.input, encode_model.layers[-2].output)
WIDTH = 299
HEIGHT = 299
OUTPUT_DIM = 2048

max_length = 34
with open('wordtoidx.txt') as f:
    data = f.read()
wordtoidx = ast.literal_eval(data)

with open('idxtoword.txt') as f:
    data = f.read()
idxtoword = ast.literal_eval(data)

vocab_size = len(idxtoword) + 1
embedding_dim = 200

def generateCaption(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idxtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def extract_features(img):
    img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = encode_model.predict(x)
    x = np.reshape(x, OUTPUT_DIM)
    return x

inputs1 = Input(shape=(OUTPUT_DIM,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
caption_model = Model(inputs=[inputs1, inputs2], outputs=outputs)

caption_model.load_weights('caption-model.hdf5')
img_file = Image.open('Sample_Image2.jpg')
img = extract_features(img_file).reshape((1,OUTPUT_DIM))

print("Caption:",generateCaption(img))