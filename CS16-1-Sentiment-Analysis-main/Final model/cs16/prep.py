import os
import cv2
import numpy as np
import pandas as pd
import keras
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split



def preprocess_images(df, folder_path, imagesize=64):

    image_data = []
    image_label = []
    polarity = []

    for i in df['ID']:
        file_path = folder_path + str(i) + '.jpg'
        try:
            if os.path.isfile(file_path):
                img = cv2.imread(file_path)
                lab = df.loc[df['ID'] == i, 'image_polarity'].values[0]
                polar = df.loc[df['ID'] == i, 'polarity'].values[0]
                img = cv2.resize(img, (imagesize, imagesize))
                image_data.append(img)
                image_label.append(lab)
                polarity.append(polar)
            else:
                print(f"{i}.jpg not found")
        except Exception as e:
                print(f"{i}.jpg not found")

    image_data = np.array(image_data)
    image_data = image_data.astype('float32') / 255

    return image_data, polarity



def preprocess_text(df, label = 'text_polarity'):
    max_words=10000
    max_len=100
    
    # Preprocess the tweet data
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(df['tweet'])
    sequences = tokenizer.texts_to_sequences(df['tweet'])
    X_text = pad_sequences(sequences, maxlen=max_len)
    y_text = df[label]
    
    return X_text, y_text


def split_data(X, y, test_size=0.2, random_state=42):
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Split testing set into validation and testing sets
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.5, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test